import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from landmarks.main_processor import get_landmarks, save_json, save_csv


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

        print(f"\n--- Processing {base_name} ---")

        try:
            coords = get_landmarks(lib, model_type, video_path, output_video_dir, output_video_name)

            if export_json_var.get():
                save_json(coords, output_json_dir, output_json_name)
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


if __name__ == "__main__":
    run_landmarks_gui()