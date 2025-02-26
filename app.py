import os
import cv2
import typer
import supervision as sv
from ultralytics import YOLO
import google.generativeai as genai
import textwrap
import pyresearch

# Set up API Key for Gemini
os.environ["API_KEY"] = ""
genai.configure(api_key=os.environ["API_KEY"])

# Initialize Gemini Model
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Load the YOLO model for object detection
yolo_model = YOLO("tree.pt")

# Define the Typer app
app = typer.Typer()

# Process webcam or video file with YOLO and Gemini
def process_webcam(output_file="output.mp4", frame_rate=10):
    cap = cv2.VideoCapture("demo2.mp4")  # Replace with 0 for the default webcam
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Resize video dimensions
    target_width = 840
    target_height = 440

    # Get the original width, height, and fps of the input video
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object for resized video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
    out = cv2.VideoWriter(output_file, fourcc, fps, (target_width, target_height))

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    frame_count = 0
    analysis_text = "Analyzing video..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 640x440
        resized_frame = cv2.resize(frame, (target_width, target_height))

        # Perform object detection
        results = yolo_model(resized_frame)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Annotate the frame with bounding boxes and labels
        annotated_frame = bounding_box_annotator.annotate(scene=resized_frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

        # Analyze every nth frame with Gemini
        if frame_count % frame_rate == 0:
            try:
                # Save the frame temporarily
                temp_image_path = "temp_frame.jpg"
                cv2.imwrite(temp_image_path, annotated_frame)

                # Upload the temporary image file
                image_file = genai.upload_file(temp_image_path)

                # Prompt for analyzing the video
                prompt = [image_file, "\n\nDescribe what you see in this video."]

                # Generate the response
                response = gemini_model.generate_content(prompt)
                analysis_text = response.text.strip()

                # Delete the temporary image file
                os.remove(temp_image_path)

            except Exception as e:
                print(f"Error during Gemini analysis: {e}")
                analysis_text = "Analysis failed."

        # Add a blurred background for the text
        text_background_height = 150  # Increase height to accommodate multiple lines
        text_background = annotated_frame[-text_background_height:, :].copy()  # Extract bottom part of the frame
        text_background = cv2.GaussianBlur(text_background, (25, 25), 0)  # Apply Gaussian blur
        annotated_frame[-text_background_height:, :] = text_background  # Overlay blurred background

        # Wrap the analysis text into multiple lines
        wrapped_text = textwrap.wrap(analysis_text, width=50)  # Adjust width based on frame size
        font_scale = 0.5  # Smaller font size for better fit
        font_thickness = 1
        text_color = (255, 255, 0)  # Green color

        # Display each line of the wrapped text
        for i, line in enumerate(wrapped_text):
            text_position = (10, target_height - text_background_height + 20 + i * 20)  # Position each line
            cv2.putText(
                annotated_frame,
                line,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness
            )

        # Write the annotated frame to the output file
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("Webcam", annotated_frame)

        frame_count += 1

        # Exit loop if 'q' is pressed
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

@app.command()
def webcam(output_file: str = "output.mp4", frame_rate: int = 10):
    typer.echo("Starting webcam processing...")
    process_webcam(output_file, frame_rate)

if __name__ == "__main__":
    app()