import base64
import json
import os
import sys
import argparse
from openai import OpenAI


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multimodal demo client for Dynamo backend")
    parser.add_argument(
        "--srv_addr",
        type=str,
        default="localhost",
        help="Server address for OpenAI API (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--image-file",
        type=str,
        nargs='+',
        dest="image_files",
        help="Path to one or more JPEG image files"
    )
    parser.add_argument(
        "--video-file",
        type=str,
        nargs='+',
        dest="video_files",
        help="Path to one or more H264 video files"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for inference (e.g., Qwen/Qwen2.5-VL-3B-Instruct)"
    )
    args = parser.parse_args()
    
    # Validate that at least one media file is provided
    if not args.image_files and not args.video_files:
        print("Error: At least one --image-file or --video-file must be provided.")
        sys.exit(1)

    # Create request content list
    req_content = []

    # Determine prompt based on what media types are provided
    has_images = args.image_files is not None
    has_videos = args.video_files is not None
    
    if has_images and has_videos:
        prompt_text = "Give short description of the images and videos, couple sentences."
    elif has_images:
        prompt_text = "Give short description of the image, couple sentences."
    else:
        prompt_text = "Give short description of the video, couple sentences."
    
    # Add text prompt
    req_content.append({
        "type": "text",
        "text": prompt_text
    })

    # Add images
    if args.image_files:
        for image_path in args.image_files:
            if not os.path.exists(image_path):
                print(f"Error: Image file not found: {image_path}")
                sys.exit(1)
            
            with open(image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read())
            
            encoded_image_text = encoded_image.decode("utf-8")
            
            req_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image_text}"
                }
            })

    # Add videos
    if args.video_files:
        for video_path in args.video_files:
            if not os.path.exists(video_path):
                print(f"Error: Video file not found: {video_path}")
                sys.exit(1)
            
            with open(video_path, "rb") as f:
                encoded_video = base64.b64encode(f.read())
            
            encoded_video_text = encoded_video.decode("utf-8")
            
            req_content.append({
                "type": "video_url",
                "video_url": {
                    "url": f"data:video/h264;base64,{encoded_video_text}"
                }
            })

    # Send request to LLM
    openai_api_key = "EMPTY"
    srv_addr = f"http://{args.srv_addr}:8000/v1"   

    client = OpenAI(
        api_key=openai_api_key,
        base_url=srv_addr,
    )

    chat_response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": req_content
            },
        ],
    )

    print("Chat response:")
    print(json.dumps(chat_response.model_dump(), indent=2))


if __name__ == "__main__":
    main()
