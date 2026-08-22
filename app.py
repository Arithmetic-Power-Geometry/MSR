import gradio as gr
from src.msr.experiment import run_single_case, METHODS, DATASETS


def run_lab(dataset, seed, n_edits, beta, margin, methods):
    if not methods:
        methods = METHODS
    df = run_single_case(dataset, int(seed), int(n_edits), float(beta), float(margin), methods)
    cols = ["method","edit_success","preservation","preserve_accuracy","logit_drift","preservation_kl","relative_param_change","structural_cost","runtime_s"]
    return df[cols]


def build_app():
    with gr.Blocks(title="Minimum Structural Repair Lab") as demo:
        gr.Markdown("# Minimum Structural Repair Lab\nChange the repair geometry and inspect edit efficacy, preservation, drift, parameter motion, and structural cost.")
        with gr.Row():
            with gr.Column(scale=1):
                dataset = gr.Dropdown(DATASETS, value="digits", label="Dataset")
                seed = gr.Number(value=7, precision=0, label="Seed")
                n_edits = gr.Slider(1, 15, value=5, step=1, label="Simultaneous edits")
                beta = gr.Slider(0, 64, value=8, step=1, label="MSR sensitivity weight beta")
                margin = gr.Slider(0.1, 2.0, value=0.75, step=0.05, label="Target margin")
                methods = gr.CheckboxGroup(METHODS, value=METHODS, label="Methods")
                run = gr.Button("Run experiment", variant="primary")
            with gr.Column(scale=2):
                gr.Markdown("**Interpretation.** Euclidean-MSR minimizes raw local parameter movement. MSR changes the geometry so movement through preservation-sensitive directions is more expensive. A useful repair should satisfy the requested edit while limiting collateral behavior change.")
                output = gr.Dataframe(label="Results", interactive=False)
        run.click(run_lab, [dataset, seed, n_edits, beta, margin, methods], output)
    return demo


app = build_app()

if __name__ == "__main__":
    app.launch()
