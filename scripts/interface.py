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
        layout=widgets.Layout(display="none")  # hidden by default
    )
    run_button = widgets.Button(description="Run", button_style="success")
    output = widgets.Output()

    # 🔹 show file box only when classify is selected
    def on_operation_change(change):
        if change["new"] == "classify":
            file_input.layout.display = "block"
        else:
            file_input.layout.display = "none"
    operation_dropdown.observe(on_operation_change, names="value")

    def on_run_clicked(_):
        clear_output(wait=False)
        with output:
            op = operation_dropdown.value
            t  = test_dropdown.value
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
