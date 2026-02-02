# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import AsyncIterator

from transformers import AutoImageProcessor, AutoTokenizer
from transformers.models.qwen2_vl import Qwen2VLVideoProcessor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.multimodal.utils import MediaConnector

import dynamo.nixl_connect as connect
from dynamo.runtime import Client, DistributedRuntime

from ..multimodal_utils import (
    ImageLoader,
    MyRequestOutput,
    encode_image_embeddings,
    encode_video_embeddings,
    get_encoder_components,
    load_vision_model,
    load_vision_model_venc_only,
    vLLMMultimodalRequest,
)

logger = logging.getLogger(__name__)

try:
    import cupy as array_module

    if not array_module.cuda.is_available():
        raise ImportError("CUDA is not available.")
    DEVICE = "cuda"
    logger.info("Using cupy for array operations (GPU mode).")
except ImportError as e:
    logger.warning(f"Failed to import cupy, falling back to numpy: {e}.")
    import numpy as array_module

    DEVICE = "cpu"

CACHE_SIZE_MAXIMUM = 8


class EncodeWorkerHandler:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        pd_worker_client: Client,
    ) -> None:
        self.pd_worker_client = pd_worker_client
        self.engine_args = engine_args
        self.model = self.engine_args.model

        self.image_loader = ImageLoader(cache_size=CACHE_SIZE_MAXIMUM)
        self.media_connector = MediaConnector(media_io_kwargs={"video": {"num_frames": 100}})
        self.image_processor = AutoImageProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )
        self.video_processor = Qwen2VLVideoProcessor.from_pretrained(
            self.model, trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, trust_remote_code=True
        )
        self.vision_model = load_vision_model_venc_only(self.model)
        self.min_workers = 1

        # Get encoder components for the model
        self.vision_encoder, self.projector = get_encoder_components(
            self.model, self.vision_model
        )
        self._connector = None

    def cleanup(self):
        pass

    async def async_init(self, runtime: DistributedRuntime):
        """Initialize the connector for RDMA transfers"""
        logger.info("Encode worker startup started.")
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()
        await self._connector.initialize()
        logger.info("Encode worker startup completed.")

    async def generate(
        self, request: vLLMMultimodalRequest, context
    ) -> AsyncIterator[str]:
        logger.debug(f"Got raw request: {request}")
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received encode request: {{ id: {request.request_id} }}.")

        request_id = request.request_id
        token_ids = request.engine_prompt["prompt_token_ids"]

        # The following steps encode the requested image and provided useful embeddings.
        # 1. Open the image from the provided URL.
        # 2. Process the image using the image processor.
        # 3. Run the image through the vision model's vision tower.
        # 4. Run the results of the vision tower through the multi-modal projector.
        # 5. Create a descriptor for the embeddings.
        # 6. Create a write operation using the serialized request and the descriptor.
        # 7. Await for the write operation to complete.
        # 8. Yield the encode response.

        try:
            video_grid_thw = None
            image_grid_thw = None

            if request.multimodal_input.image_url:
                image = await self.image_loader.load_image(
                    request.multimodal_input.image_url
                )

                logger.debug(f"Processing image for request: {{ id: {request_id} }}")
                image_embeds = self.image_processor(images=image, return_tensors="pt")

                # Encode the image embeddings using model-specific encoder
                embeddings = encode_image_embeddings(
                    model_name=self.model,
                    image_embeds=image_embeds,
                    vision_encoder=self.vision_encoder,
                    projector=self.projector,
                )

                image_grid_thw = (
                    image_embeds["image_grid_thw"].tolist()
                    if "image_grid_thw" in image_embeds
                    else None
                )
                logger.debug(
                    f"Pixel values stats: mean={image_embeds['pixel_values'].mean().item()}, std={image_embeds['pixel_values'].std().item()}, min={image_embeds['pixel_values'].min().item()}, max={image_embeds['pixel_values'].max().item()}"
                )

            elif request.multimodal_input.video_url:
                image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                video_pad_id = self.tokenizer.convert_tokens_to_ids("<|video_pad|>")
                token_ids = [video_pad_id if t == image_pad_id else t for t in token_ids]
                request.engine_prompt["prompt_token_ids"] = token_ids

                video_url = request.multimodal_input.video_url
                video, video_metadata = await self.media_connector.fetch_video_async(video_url)

                # Process video using Qwen2VLVideoProcessor
                video_embeds = self.video_processor(videos=[video], return_tensors="pt")

                # Encode the video embeddings using model-specific encoder
                embeddings = encode_video_embeddings(
                    model_name=self.model,
                    video_embeds=video_embeds,
                    vision_encoder=self.vision_encoder,
                    projector=self.projector,
                )

                video_grid_thw = (
                    video_embeds["video_grid_thw"].tolist()
                    if "video_grid_thw" in video_embeds
                    else None
                )

            else:
                raise ValueError("image_url or video_url is required for the encode worker.")

            # Move embeddings to CPU for NIXL transfer to avoid UCX/InfiniBand issues
            embeddings_cpu = embeddings.cpu()

            request.image_grid_thw = image_grid_thw
            request.video_grid_thw = video_grid_thw
            request.embeddings_shape = tuple(embeddings.shape)
            descriptor = connect.Descriptor(embeddings_cpu)

            with self._connector.create_readable(descriptor) as readable:
                request.serialized_request = readable.metadata()
                # Clear the image URL as hint that the image is passed as embeddings.
                request.multimodal_input.image_url = None
                request.multimodal_input.video_url = None

                logger.debug(f"Request: {request.model_dump_json()}")

                # Get the response generator
                response_generator = await self.pd_worker_client.round_robin(
                    request.model_dump_json(), context=context
                )
                await readable.wait_for_completion()

                async for response in response_generator:
                    output = MyRequestOutput.model_validate_json(response.data())
                    yield MyRequestOutput(
                        request_id=output.request_id,
                        prompt=output.prompt,
                        prompt_token_ids=output.prompt_token_ids,
                        prompt_logprobs=output.prompt_logprobs,
                        outputs=output.outputs,
                        finished=output.finished,
                    ).model_dump_json()

        except Exception as e:
            logger.error(f"Error processing request {request_id}: {e}")
            raise
