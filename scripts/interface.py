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

    # hidden by default
    file_input = widgets.Text(
        description="Video file:",
        placeholder="example.mov",
    )
    file_input.layout.visibility = "hidden"

    run_button = widgets.Button(description="Run", button_style="success")
    output = widgets.Output()

    # 🔹 Hide/show the file input dynamically
    def on_operation_change(change):
        if change["new"] == "classify":
            file_input.layout.visibility = "visible"
        else:
            file_input.layout.visibility = "hidden"

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

    ui = widgets.VBox([
        widgets.HTML("<h3 style='color:#2c3e50;'>🧠 Neuro-Exam Automation Interface</h3>"),
        operation_dropdown,
        test_dropdown,
        file_input,
        run_button,
        output
    ])

    display(ui)
    # ensure initial state is hidden
    file_input.layout.visibility = "hidden"
