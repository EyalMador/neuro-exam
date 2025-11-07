import ipywidgets as widgets
from IPython.display import display, clear_output
from scripts.automation import train, classify_video, test

supported_tests = ["straight_walk", "heel_to_toe_walk", "raise_hands", "finger_to_nose"]

def main():
    op_dd = widgets.Dropdown(
        options=["train", "classify", "test"],
        description="Operation:",
    )
    test_dd = widgets.Dropdown(
        options=supported_tests,
        description="Test:",
    )
    file_box = widgets.Text(description="Video file:", placeholder="example.mov")
    run_btn = widgets.Button(description="Run", button_style="success")
    out = widgets.Output()

    # hidden initially
    file_box.layout.display = "none"

    def on_op_change(change):
        file_box.layout.display = "block" if change["new"] == "classify" else "none"
    op_dd.observe(on_op_change, names="value")

    def on_run(_):
        clear_output(wait=False)
        with out:
            op, t = op_dd.value, test_dd.value
            if op == "train":
                train(t)
            elif op == "test":
                test(t)
            elif op == "classify":
                classify_video(t, file_box.value)

    run_btn.on_click(on_run)

    ui = widgets.VBox([
        widgets.HTML("<h3>🧠 Neuro-Exam Automation Interface</h3>"),
        op_dd, test_dd, file_box, run_btn, out
    ])
    display(ui)
