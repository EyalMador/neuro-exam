import ipywidgets as widgets
from IPython.display import display, clear_output
from scripts.automation import train, classify_video, test

supported_tests = ["straight_walk", "heel_to_toe_walk", "raise_hands", "finger_to_nose"]

def main():
    operation_dropdown = widgets.Dropdown(
        options=["train", "classify", "test"],
        description="Operation:",
    )

    test_dropdown = widgets.Dropdown(
        options=supported_tests,
        description="Test:",
    )

    file_input = widgets.Text(
        description="Video file:",
        placeholder="example.mov",
    )

    run_button = widgets.Button(description="Run", button_style="success")
    output = widgets.Output()

    # Container that will hold widgets dynamically
    dynamic_box = widgets.VBox([operation_dropdown, test_dropdown, run_button, output])

    def refresh_ui():
        # remove or add the file input based on current operation
        children = [operation_dropdown, test_dropdown]
        if operation_dropdown.value == "classify":
            children.append(file_input)
        children += [run_button, output]
        dynamic_box.children = children

    def on_operation_change(change):
        refresh_ui()

    operation_dropdown.observe(on_operation_change, names="value")

    def on_run_clicked(_):
        clear_output(wait=False)
        with output:
            op = operation_dropdown.value
            t = test_dropdown.value
            if op == "train":
                train(t)
            elif op == "test":
                test(t)
            elif op == "classify":
                classify_video(t, file_input.value)

    run_button.on_click(on_run_clicked)

    # UI header
    title = widgets.HTML("<h3 style='color:#2c3e50;'>🧠 Neuro-Exam Automation Interface</h3>")
    ui = widgets.VBox([title, dynamic_box])

    # initial layout (without file input)
    refresh_ui()
    display(ui)
