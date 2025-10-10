import os
import json
import csv
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# --- Import your biomarker functions ---
from biomarkers.finger_to_nose import extract_finger_to_nose_biomarkers
from biomarkers.straight_walk import extract_straight_walk_biomarkers

BIOMARKERS = {
    "Finger to Nose": extract_finger_to_nose_biomarkers,
    "Straight Walk": extract_straight_walk_biomarkers,
}


def run_biomarkers_gui():
    root = tk.Tk()
    root.title("Run Biomarker Extraction")
    root.geometry("480x400")
    root.configure(bg="white")

    # ---- Variables ----
    json_path_var = tk.StringVar()
    biomarker_var = tk.StringVar(value=list(BIOMARKERS.keys())[0])
    output_file_var = tk.StringVar()

    export_json_var = tk.BooleanVar(value=True)
    export_csv_var = tk.BooleanVar(value=True)

    # ---- Helper functions ----
    def choose_json():
        path = filedialog.askopenfilename(
            title="Select Landmarks JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            json_path_var.set(path)

    def choose_output_file():
        file_path = filedialog.asksaveasfilename(
            title="Save Biomarker Output As",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            output_file_var.set(file_path)

    def save_results(results, base_path):
        """Save results as JSON and/or CSV."""
        base_dir = os.path.dirname(base_path)
        base_name = os.path.splitext(os.path.basename(base_path))[0]

        if export_json_var.get():
            json_out = os.path.join(base_dir, f"{base_name}.json")
            with open(json_out, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Saved JSON: {json_out}")

        if export_csv_var.get():
            csv_out = os.path.join(base_dir, f"{base_name}.csv")
            if isinstance(results, dict):
                with open(csv_out, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["metric", "value"])
                    for k, v in results.items():
                        writer.writerow([k, v])
                print(f"Saved CSV: {csv_out}")
            else:
                print("Cannot save CSV — results not a dictionary.")

    def run_biomarker():
        json_path = json_path_var.get().strip()
        biomarker_name = biomarker_var.get()
        output_path = output_file_var.get().strip()

        if not os.path.exists(json_path):
            messagebox.showerror("Error", "Invalid JSON file selected.")
            return
        if not output_path:
            messagebox.showerror("Error", "Please choose save location and name.")
            return

        with open(json_path, "r") as f:
            coords = json.load(f)

        func = BIOMARKERS[biomarker_name]
        try:
            results = func(coords)
        except Exception as e:
            messagebox.showerror("Error", f"Error while running {biomarker_name}:\n{e}")
            raise

        save_results(results, output_path)
        messagebox.showinfo("Success", f"✅ {biomarker_name} biomarker completed!\nResults saved to:\n{output_path}")

    # ---- UI Layout ----
    pad_y = 5
    label_style = {"bg": "white", "font": ("Helvetica", 10)}

    tk.Label(root, text="Select Biomarker:", **label_style).pack(pady=pad_y)
    ttk.Combobox(root, textvariable=biomarker_var, values=list(BIOMARKERS.keys()),
                 state="readonly", width=25).pack()

    tk.Label(root, text="Landmarks JSON File:", **label_style).pack(pady=pad_y)
    tk.Entry(root, textvariable=json_path_var, width=55, relief="solid", borderwidth=1).pack()
    tk.Button(root, text="Browse JSON", command=choose_json,
              width=20, bg="#FAFAFA", fg="black", relief="flat").pack(pady=pad_y)

    tk.Label(root, text="Output File:", **label_style).pack(pady=pad_y)
    tk.Entry(root, textvariable=output_file_var, width=55, relief="solid", borderwidth=1).pack()
    tk.Button(root, text="Choose Save Location", command=choose_output_file,
              width=20, bg="#FAFAFA", fg="black", relief="flat").pack(pady=pad_y)

    tk.Label(root, text="Export Options:", **label_style).pack(pady=pad_y)
    tk.Checkbutton(root, text="Export JSON", variable=export_json_var, bg="white").pack()
    tk.Checkbutton(root, text="Export CSV", variable=export_csv_var, bg="white").pack()

    tk.Button(
        root,
        text="Run Biomarker",
        command=run_biomarker,
        bg="#C8E6C9",  # light green
        fg="black",
        activebackground="#B2DFDB",
        activeforeground="black",
        relief="flat",
        width=22,
        height=2,
        font=("Helvetica", 11, "bold")
    ).pack(pady=20)

    root.mainloop()


if __name__ == "__main__":
    run_biomarkers_gui()