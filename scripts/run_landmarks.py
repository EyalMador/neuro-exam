import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from landmarks.main_processor import get_landmarks, save_json, save_csv
from pathlib import Path
import cv2


def run_landmarks_batch(
    input_dir,
    output_dir,
    lib="mediapipe",
    model_type="pose",
    export_video=True,
    export_json=True,
    export_csv=True,
    video_extensions=None
):
    """
    Batch process multiple videos from a directory.
    
    Args:
        input_dir (str): Path to directory containing video files
        output_dir (str): Path to directory where outputs will be saved
        lib (str): Library to use - "mediapipe" or "rtmlib"
        model_type (str): Model type - "pose", "hands", "holistic" (mediapipe) or "body26" (rtmlib)
        export_video (bool): Whether to export annotated video
        export_json (bool): Whether to export JSON landmarks
        export_csv (bool): Whether to export CSV landmarks
        video_extensions (list): List of video extensions to process
    
    Returns:
        dict: Results summary with successful and failed videos
    
    Example:
        results = run_landmarks_batch(
            input_dir="/path/to/input/videos",
            output_dir="/path/to/output",
            lib="mediapipe",
            model_type="pose",
            export_video=True,
            export_json=True,
            export_csv=True
        )
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV']
    
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Create output directories
    output_video_dir = os.path.join(output_dir, "Video_output")
    output_json_dir = Path(output_dir)
    os.makedirs(output_video_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)
    
    # Get all video files
    input_path = Path(input_dir)
    video_files = [f for f in input_path.iterdir() 
                   if f.is_file() and f.suffix in video_extensions]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        print(f"Looking for extensions: {video_extensions}")
        return {"successful": [], "failed": []}
    
    print(f"\nFound {len(video_files)} video(s) to process")
    print(f"Library: {lib}, Model: {model_type}")
    print(f"Export options - Video: {export_video}, JSON: {export_json}, CSV: {export_csv}\n")
    
    results = {"successful": [], "failed": []}
    
    # Process each video
    for idx, video_file in enumerate(video_files, 1):
        video_path = str(video_file)
        base_name = video_file.stem
        
        output_video_name = f"{base_name}.mp4"
        output_json_name = f"{base_name}.json"
        output_csv_name = f"{base_name}.csv"

        #get video metadata:
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\n[{idx}/{len(video_files)}] Processing: {base_name}")
        
        try:
            coords = get_landmarks(lib, model_type, video_path, output_video_dir, output_video_name)
            
            if export_json:
                save_json(coords, output_json_dir, output_json_name, frame_width, frame_height)
                print(f"  Saved JSON: {output_json_name}")
            
            if export_csv:
                save_csv(coords, output_json_dir, output_csv_name)
                print(f"  Saved CSV: {output_csv_name}")
            
            if not export_video:
                video_path_generated = os.path.join(output_video_dir, output_video_name)
                if os.path.exists(video_path_generated):
                    os.remove(video_path_generated)
            else:
                print(f"  Saved Video: {output_video_name}")
            
            results["successful"].append(base_name)
            print(f"  Successfully processed {base_name}")
            
        except Exception as e:
            print(f"  Failed to process {base_name}: {str(e)}")
            results["failed"].append({"file": base_name, "error": str(e)})
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Successful: {len(results['successful'])}/{len(video_files)}")
    print(f"Failed: {len(results['failed'])}/{len(video_files)}")
    
    if results['failed']:
        print("\nFailed files:")
        for failed in results['failed']:
            print(f"  - {failed['file']}: {failed['error']}")
    
    print(f"\nOutputs saved to: {output_dir}")
    print("="*60 + "\n")
    
    return results

def run_landmarks_gui():
    root = tk.Tk()
    root.title("Run Landmarks Extraction")
    root.geometry("500x500")  # Increased height
    root.configure(bg="white")

    # ---- Variables ----
    lib_var = tk.StringVar(value="mediapipe")
    model_type_var = tk.StringVar(value="pose")
    video_path_var = tk.StringVar()
    output_file_var = tk.StringVar()

    export_video_var = tk.BooleanVar(value=True)
    export_json_var = tk.BooleanVar(value=True)
    export_csv_var = tk.BooleanVar(value=True)

    # ---- Helper Functions ----
    def choose_video():
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if path:
            video_path_var.set(path)

    def choose_output_file():
        file_path = filedialog.asksaveasfilename(
            title="Save Output File As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            output_file_var.set(file_path)

    def update_model_types(*args):
        lib = lib_var.get()
        if lib == "mediapipe":
            model_combo["values"] = ["pose", "hands", "holistic"]
            if model_type_var.get() not in model_combo["values"]:
                model_type_var.set("pose")
        elif lib == "rtmlib":
            model_combo["values"] = ["body26"]
            model_type_var.set("body26")

    def run_extraction():
        video_path = video_path_var.get().strip()
        output_path = output_file_var.get().strip()
        lib = lib_var.get()
        model_type = model_type_var.get()

        if not os.path.exists(video_path):
            messagebox.showerror("Error", "Invalid video file selected.")
            return
        if not output_path:
            messagebox.showerror("Error", "Please choose output file name and folder.")
            return

        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]

        output_video_dir = os.path.join(output_dir, "Video_output")
        output_json_dir = os.path.join(output_dir, "Landmarks_output")
        os.makedirs(output_video_dir, exist_ok=True)
        os.makedirs(output_json_dir, exist_ok=True)

        output_video_name = f"{base_name}.mp4"
        output_json_name = f"{base_name}.json"
        output_csv_name = f"{base_name}.csv"

        #get video metadata:
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n--- Processing {base_name} ---")

        try:
            coords = get_landmarks(lib, model_type, video_path, output_video_dir, output_video_name)

            if export_json_var.get():
                save_json(coords, output_json_dir, output_json_name, frame_width, frame_height)
            if export_csv_var.get():
                save_csv(coords, output_json_dir, output_csv_name)

            if not export_video_var.get():
                video_path_generated = os.path.join(output_video_dir, output_video_name)
                if os.path.exists(video_path_generated):
                    os.remove(video_path_generated)

            messagebox.showinfo("Success", f"Extraction complete!\nSaved to:\n{output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract landmarks:\n{e}")
            raise

    # ---- UI Layout ----
    pad_y = 8
    label_style = {"bg": "white", "font": ("Helvetica", 10)}

    tk.Label(root, text="Select Library:", **label_style).pack(pady=(10, 2))
    lib_combo = ttk.Combobox(root, textvariable=lib_var,
                             values=["mediapipe", "rtmlib"],
                             state="readonly", width=20)
    lib_combo.pack(pady=(0, pad_y))
    lib_combo.bind("<<ComboboxSelected>>", update_model_types)

    tk.Label(root, text="Select Model Type:", **label_style).pack(pady=(5, 2))
    model_combo = ttk.Combobox(root, textvariable=model_type_var,
                               values=["pose", "hands", "holistic"],
                               state="readonly", width=20)
    model_combo.pack(pady=(0, pad_y))

    tk.Label(root, text="Video File:", **label_style).pack(pady=(5, 2))
    tk.Entry(root, textvariable=video_path_var, width=55,
             relief="solid", borderwidth=1).pack(pady=(0, 2))
    tk.Button(root, text="Browse Video", command=choose_video,
              width=20, bg="#FAFAFA", fg="black", relief="flat").pack(pady=(0, pad_y))

    tk.Label(root, text="Output File:", **label_style).pack(pady=(5, 2))
    tk.Entry(root, textvariable=output_file_var, width=55,
             relief="solid", borderwidth=1).pack(pady=(0, 2))
    tk.Button(root, text="Choose Save Location", command=choose_output_file,
              width=20, bg="#FAFAFA", fg="black", relief="flat").pack(pady=(0, pad_y))

    tk.Label(root, text="Export Options:", **label_style).pack(pady=(5, 2))
    export_frame = tk.Frame(root, bg="white")
    export_frame.pack(pady=(0, 10))
    tk.Checkbutton(export_frame, text="Export Video", variable=export_video_var, bg="white").pack()
    tk.Checkbutton(export_frame, text="Export JSON", variable=export_json_var, bg="white").pack()
    tk.Checkbutton(export_frame, text="Export CSV", variable=export_csv_var, bg="white").pack()

    # --- Run Extraction Button (with spacing) ---
    tk.Label(root, text="", bg="white").pack(pady=5)  # Spacer
    
    run_button = tk.Button(
        root,
        text="Run Extraction",
        command=run_extraction,
        bg="#4CAF50",  # Material green
        fg="white",
        activebackground="#45A049",
        activeforeground="white",
        relief="raised",
        width=22,
        height=2,
        font=("Helvetica", 12, "bold"),
        cursor="hand2"
    )
    run_button.pack(pady=(10, 20))

    root.mainloop()

def run_extraction_with_args(video_path,output_path, lib, model_type, base_name, output_video_dir):
    video_name = base_name.split('.', 1)[0]
    output_json_name = f"{video_name}.json"
    
    print(f"\n--- Processing {base_name} ---")

    try:
        coords = get_landmarks(lib, model_type, video_path, output_video_dir, base_name)
        save_json(coords, output_path, output_json_name)
    except Exception as e:
        raise

if __name__ == "__main__":
    run_landmarks_gui()
