import inspect
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os, json
from pprint import pformat
import matplotlib.pyplot as plt

# --- Example imports ---
from biomarkers.finger_to_nose import extract_finger_to_nose_biomarkers
from biomarkers.straight_walk import extract_straight_walk_biomarkers
from biomarkers.heel_to_toe_walk import extract_heel_to_toe_biomarkers


BIOMARKERS = {
    # "Finger to Nose": extract_finger_to_nose_biomarkers,
    "Straight Walk": extract_straight_walk_biomarkers,
    "Heel To Toe": extract_heel_to_toe_biomarkers
}



def biomarkers_to_json_format(biomarkers, result="normal"):
    """
    Convert biomarkers dictionary to standardized JSON format.
    
    Parameters:
    -----------
    biomarkers : dict
        Dictionary containing biomarker categories with statistics
        Example: {
            "heel_toe_distances": {
                "left": {"mean": 10.5, "std": 2.3, ...},
                "right": {"mean": 11.2, "std": 2.1, ...}
            },
            "knee_angles": {
                "left": {"mean": 45.2, "std": 5.1, ...},
                "right": {"mean": 44.8, "std": 4.9, ...}
            }
        }
    result : str
        Classification result (e.g., "normal", "abnormal")
    
    Returns:
    --------
    dict : Formatted dictionary ready for JSON export
    """
    formatted = {
        "biomarkers": {},
        "result": result
    }
    
    biomarker_counter = 1
    
    for category_name, category_data in biomarkers.items():
        if not isinstance(category_data, dict):
            continue
        
        # Handle different nesting structures
        for key, value in category_data.items():
            # Skip non-statistical keys
            if key in ['all', 'all_distances', 'min_frames', 'min_values', 
                      'min_indices', 'smoothed_distances', 'properties', 'count']:
                continue
            
            # If value is a dict with mean/std, extract it
            if isinstance(value, dict) and ('mean' in value or 'std' in value):
                biomarker_name = f"b{biomarker_counter}"
                formatted["biomarkers"][biomarker_name] = {
                    "name": f"{category_name}_{key}",
                    "mean": round(value.get('mean', 0), 2),
                    "std": round(value.get('std', 0), 2)
                }
                biomarker_counter += 1
    
    return formatted


def save_biomarkers_json(biomarkers, output_dir, filename, result="normal"):
    """
    Save biomarkers to JSON file in standardized format.
    
    Parameters:
    -----------
    biomarkers : dict
        Biomarkers dictionary from extract_*_biomarkers functions
    output_dir : str
        Directory to save the JSON file
    filename : str
        Name of the output file (with or without .json extension)
    result : str
        Classification result
    
    Returns:
    --------
    str : Path to saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not filename.endswith('.json'):
        filename += '.json'
    
    output_path = os.path.join(output_dir, filename)
    
    formatted_data = biomarkers_to_json_format(biomarkers, result)
    
    with open(output_path, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Biomarkers saved to: {output_path}")
    return output_path


def run_biomarkers_gui():
    root = tk.Tk()
    root.title("Run Biomarker Extraction")
    root.geometry("600x650")
    root.configure(bg="white")

    json_path_var = tk.StringVar()
    biomarker_var = tk.StringVar(value=list(BIOMARKERS.keys())[0])
    param_entries = {}

    # --- Functions ---
    def choose_json():
        path = filedialog.askopenfilename(
            title="Select Landmarks JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if path:
            json_path_var.set(path)

    def visualize_results(results):
        """If results contain numeric lists, plot them."""
        numeric_dict = {k: v for k, v in results.items() if isinstance(v, (list, tuple))}
        if not numeric_dict:
            return
        plt.figure(figsize=(6, 4))
        for key, values in numeric_dict.items():
            try:
                plt.plot(values, label=key)
            except Exception:
                pass
        plt.title("Biomarker Visualization")
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    def update_param_fields(*_):
        """Show input fields for detected parameters (except coords/landmarks)."""
        for widget in params_frame.winfo_children():
            widget.destroy()
        param_entries.clear()

        func = BIOMARKERS[biomarker_var.get()]
        sig = inspect.signature(func)

        tk.Label(params_frame, text="Optional Parameters:", bg="white",
                 font=("Helvetica", 10, "bold")).pack(pady=5)

        for name, param in sig.parameters.items():
            # skip the "coords" or "landmarks" param
            if name.lower() in ["coords", "landmarks"]:
                continue

            label_text = f"{name}"
            if param.default is not inspect._empty:
                label_text += f" (default: {param.default})"

            tk.Label(params_frame, text=label_text, bg="white", font=("Helvetica", 9)).pack()
            entry = tk.Entry(params_frame, width=30, relief="solid", borderwidth=1)
            entry.pack(pady=3)

            # prefill default if available
            if param.default is not inspect._empty:
                entry.insert(0, str(param.default))
            param_entries[name] = entry

    def run_biomarker():
        json_path = json_path_var.get().strip()
        biomarker_name = biomarker_var.get()

        if not os.path.exists(json_path):
            messagebox.showerror("Error", "Invalid JSON file selected.")
            return

        with open(json_path, "r") as f:
            coords = json.load(f)

        func = BIOMARKERS[biomarker_name]
        kwargs = {}

        # gather all user-entered parameters
        for name, entry in param_entries.items():
            val = entry.get().strip()
            if val:
                try:
                    kwargs[name] = eval(val)
                except Exception:
                    kwargs[name] = val  # keep as string if not evaluable

        try:
            results = func(coords, **kwargs)
        except Exception as e:
            messagebox.showerror("Error", f"Error while running {biomarker_name}:\n{e}")
            raise

        # print and visualize results
        result_box.config(state="normal")
        result_box.delete("1.0", tk.END)
        result_box.insert(tk.END, pformat(results, indent=2))
        result_box.config(state="disabled")

        visualize_results(results)
        print(f"✅ {biomarker_name} biomarker completed successfully.")

    # --- Layout ---
    pad_y = 5
    label_style = {"bg": "white", "font": ("Helvetica", 10)}

    tk.Label(root, text="Select Biomarker:", **label_style).pack(pady=pad_y)
    biomarker_menu = ttk.Combobox(
        root, textvariable=biomarker_var, values=list(BIOMARKERS.keys()),
        state="readonly", width=25
    )
    biomarker_menu.pack()
    biomarker_var.trace_add("write", update_param_fields)

    tk.Label(root, text="Landmarks JSON File:", **label_style).pack(pady=pad_y)
    tk.Entry(root, textvariable=json_path_var, width=55, relief="solid", borderwidth=1).pack()
    tk.Button(root, text="Browse JSON", command=choose_json,
              width=20, bg="#FAFAFA", fg="black", relief="flat").pack(pady=pad_y)

    params_frame = tk.Frame(root, bg="white")
    params_frame.pack(pady=10)
    update_param_fields()

    tk.Button(
        root,
        text="Run Biomarker",
        command=run_biomarker,
        bg="#C8E6C9",
        fg="black",
        relief="flat",
        width=22,
        height=2,
        font=("Helvetica", 11, "bold")
    ).pack(pady=10)

    tk.Label(root, text="Results:", **label_style).pack(pady=pad_y)
    global result_box
    result_box = tk.Text(root, width=70, height=15, wrap="word",
                         bg="#F9F9F9", fg="black", font=("Courier", 9))
    result_box.pack(pady=10)
    result_box.config(state="disabled")

    root.mainloop()


if __name__ == "__main__":
    run_biomarkers_gui()
