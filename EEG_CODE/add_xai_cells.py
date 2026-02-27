"""Script to add XAI analysis cells to the notebook."""
import json

# Read notebook
with open('CrossModal_V4_final_0.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Total cells before: {len(nb["cells"])}')

# Define new XAI cells
xai_cells = [
    # Cell 1: Import XAI module
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''# ============================================================================
# EXPLAINABLE AI (XAI) ANALYSIS
# ============================================================================
# Import XAI module for channel importance and brain mapping

from eeg_xai_analysis import (
    EEGExplainer,
    ChannelImportanceExtractor,
    plot_channel_importance,
    plot_topomap,
    plot_region_comparison,
    plot_connectivity_matrix,
    create_analysis_report,
    STANDARD_10_20_19,
    BRAIN_REGIONS
)

print("XAI Analysis tools loaded successfully")'''
    },

    # Cell 2: Configure channel names
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''# ============================================================================
# CONFIGURE CHANNEL NAMES FOR YOUR EEG MONTAGE
# ============================================================================
# Modify this list to match your actual EEG channel configuration
# Example for 18 channels (modify as needed):

EEG_CHANNEL_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",  # Frontal
    "T3", "C3", "Cz", "C4", "T4",                  # Central/Temporal
    "T5", "P3", "Pz", "P4", "T6",                  # Parietal/Temporal
    "O1"                                           # Occipital (add O2 if 19ch)
]

# If your channels are different, list them in order here:
# EEG_CHANNEL_NAMES = ["Ch1", "Ch2", ..., "ChN"]

print(f"Configured {len(EEG_CHANNEL_NAMES)} EEG channels: {EEG_CHANNEL_NAMES}")'''
    },

    # Cell 3: XAI analysis function
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''# ============================================================================
# RUN XAI ANALYSIS ON TRAINED MODEL
# ============================================================================

def run_xai_analysis(model, dataloader, channel_names, output_dir="xai_results",
                    model_name="model", max_samples=50):
    """
    Run comprehensive XAI analysis on trained model.

    Args:
        model: Trained model (ImprovedTriModalFusionNet or ImprovedSmartFusionNet)
        dataloader: DataLoader with test samples
        channel_names: List of EEG channel names
        output_dir: Directory to save results
        model_name: Name for saving files

    Returns:
        results_dict: Channel importance rankings for each modality
        explainer: EEGExplainer object for further analysis
    """
    from pathlib import Path
    import matplotlib.pyplot as plt

    output_path = Path(output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\\n{'='*60}")
    print(f"XAI ANALYSIS: {model_name.upper()}")
    print(f"{'='*60}")

    # Initialize explainer
    explainer = EEGExplainer(
        model=model,
        channel_names=channel_names,
        device=device
    )

    # Analyze samples
    print(f"\\nAnalyzing up to {max_samples} samples...")
    n_analyzed = 0

    for batch in dataloader:
        if n_analyzed >= max_samples:
            break

        # Handle TriModal dataset format
        if len(batch) == 5:
            erp, pw, conn, subj, labels = batch
            results = explainer.analyze_sample(erp, pw, conn, methods=["gradient", "integrated_gradients"])
        else:  # BiModal
            erp, pw, labels = batch[0], batch[1], batch[-1]
            results = explainer.analyze_sample(erp, pw, methods=["gradient", "integrated_gradients"])

        n_analyzed += erp.shape[0]

    print(f"Analyzed {n_analyzed} samples")

    # Get channel rankings
    print("\\n" + "-"*40)
    print("CHANNEL IMPORTANCE RANKINGS")
    print("-"*40)

    results_dict = {}

    for modality in ["erp", "pw"]:
        ranking = explainer.get_channel_ranking(modality=modality, method="gradient")
        results_dict[modality] = ranking

        print(f"\\n{modality.upper()} - Top 10 Most Important Channels:")
        for rank, (ch, score) in enumerate(ranking[:10], 1):
            print(f"  {rank:2d}. {ch:6s}: {score:.4f}")

        # Plot channel importance
        fig, ax = plot_channel_importance(
            dict(ranking[:15]),
            title=f"{model_name} - {modality.upper()} Channel Importance",
            save_path=str(output_path / f"{modality}_channel_importance.png")
        )
        plt.show()

        # Plot topomap
        fig, ax = plot_topomap(
            dict(ranking),
            title=f"{model_name} - {modality.upper()} Topographic Map",
            save_path=str(output_path / f"{modality}_topomap.png")
        )
        plt.show()

    # Region importance
    print("\\n" + "-"*40)
    print("BRAIN REGION IMPORTANCE")
    print("-"*40)

    ch_extractor = ChannelImportanceExtractor(channel_names=channel_names)

    for modality in ["erp", "pw"]:
        ch_imp = dict(results_dict[modality])
        region_imp = ch_extractor.get_region_importance(ch_imp)

        print(f"\\n{modality.upper()} Region Importance:")
        for region, score in sorted(region_imp.items(), key=lambda x: x[1], reverse=True):
            print(f"  {region:12s}: {score:.4f}")

        # Radar plot
        fig, ax = plot_region_comparison(
            region_imp,
            title=f"{model_name} - {modality.upper()} Region Importance",
            save_path=str(output_path / f"{modality}_region_radar.png")
        )
        plt.show()

    print(f"\\nResults saved to: {output_path}")
    return results_dict, explainer

print("XAI analysis function defined")'''
    },

    # Cell 4: Analyze tri-modal
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''# ============================================================================
# ANALYZE TRI-MODAL MODEL
# ============================================================================
# Run this after training to analyze the tri-modal fusion model

# Load best checkpoint if available
trimodal_ckpts = list(config.checkpoint_dir.glob("best_trimodal_fold*.pt"))

if trimodal_ckpts and 'trimodal_ds' in dir() and trimodal_ds is not None:
    print("Loading best tri-modal checkpoint for XAI analysis...")

    # Get dimensions from dataset
    sample_erp, sample_pw, sample_conn, _, _ = trimodal_ds[0]
    erp_ch = sample_erp.shape[0]
    pw_ch = sample_pw.shape[0]
    conn_dim = sample_conn.shape[0]

    # Create model
    trimodal_model = ImprovedTriModalFusionNet(
        in_pw_dim=pw_ch,
        in_erp_dim=erp_ch,
        in_conn_dim=conn_dim,
        fusion_dim=config.fusion_dim,
        num_classes=n_classes
    ).to(device)

    # Load weights
    ckpt = torch.load(trimodal_ckpts[0], map_location=device)
    trimodal_model.load_state_dict(ckpt["model_state_dict"])
    trimodal_model.eval()

    # Create dataloader
    xai_loader = DataLoader(trimodal_ds, batch_size=8, shuffle=False)

    # Run analysis
    trimodal_results, trimodal_explainer = run_xai_analysis(
        model=trimodal_model,
        dataloader=xai_loader,
        channel_names=EEG_CHANNEL_NAMES,
        output_dir=str(config.output_dir / "xai_analysis"),
        model_name="trimodal",
        max_samples=100
    )
else:
    print("No tri-modal checkpoint found or dataset not available. Train the model first.")'''
    },

    # Cell 5: Analyze bi-modal
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''# ============================================================================
# ANALYZE BI-MODAL MODEL
# ============================================================================
# Run this after training to analyze the bi-modal fusion model

# Load best checkpoint if available
bimodal_ckpts = list(config.checkpoint_dir.glob("best_fusion_fold*.pt"))

if bimodal_ckpts:
    print("Loading best bi-modal checkpoint for XAI analysis...")

    # Create model with same dimensions
    bimodal_model = ImprovedSmartFusionNet(
        in_pw_dim=pw_ch,
        in_erp_dim=erp_ch,
        fusion_dim=config.fusion_dim,
        num_classes=n_classes,
        use_cross_attention=True
    ).to(device)

    # Load weights
    ckpt = torch.load(bimodal_ckpts[0], map_location=device)
    bimodal_model.load_state_dict(ckpt["model_state_dict"])
    bimodal_model.eval()

    print("Bi-modal model loaded successfully")
    print("Note: Use similar XAI analysis pattern as tri-modal")
else:
    print("No bi-modal checkpoint found. Train the model first.")'''
    },

    # Cell 6: Generate summary report
    {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': '''# ============================================================================
# GENERATE FINAL XAI SUMMARY REPORT
# ============================================================================

def generate_xai_summary(results_dict, channel_names, model_name, output_path):
    """
    Generate a comprehensive text summary of XAI findings.
    """
    from pathlib import Path

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    ch_extractor = ChannelImportanceExtractor(channel_names=channel_names)

    report_lines = [
        "="*70,
        f"EEG EXPLAINABILITY REPORT: {model_name.upper()}",
        "="*70,
        "",
        f"Channel Configuration: {len(channel_names)} channels",
        f"Channels: {', '.join(channel_names)}",
        "",
    ]

    for modality in ["erp", "pw"]:
        if modality not in results_dict:
            continue

        ranking = results_dict[modality]
        ch_imp = dict(ranking)
        region_imp = ch_extractor.get_region_importance(ch_imp)

        report_lines.extend([
            "-"*50,
            f"{modality.upper()} MODALITY ANALYSIS",
            "-"*50,
            "",
            "Top 10 Most Discriminative Channels:",
        ])

        for rank, (ch, score) in enumerate(ranking[:10], 1):
            # Find brain region
            region = "Unknown"
            for reg, chs in BRAIN_REGIONS.items():
                if ch in chs:
                    region = reg
                    break
            report_lines.append(f"  {rank:2d}. {ch:6s} ({region:10s}): {score:.4f}")

        report_lines.extend([
            "",
            "Brain Region Importance (aggregated):",
        ])

        sorted_regions = sorted(region_imp.items(), key=lambda x: x[1], reverse=True)
        for region, score in sorted_regions:
            report_lines.append(f"  {region:12s}: {score:.4f}")

        # Key findings
        top_region = sorted_regions[0][0]
        top_channels = [ch for ch, _ in ranking[:3]]

        report_lines.extend([
            "",
            "Key Findings:",
            f"  - Most important brain region: {top_region}",
            f"  - Top 3 channels: {', '.join(top_channels)}",
            "",
        ])

    report_lines.extend([
        "="*70,
        "CLINICAL INTERPRETATION NOTES",
        "="*70,
        "",
        "The channel importance scores indicate which EEG electrodes",
        "contribute most to the model's classification decisions.",
        "",
        "Higher scores suggest:",
        "  - More discriminative neural activity at that location",
        "  - Potential biomarker regions for the classification task",
        "",
        "These findings can guide:",
        "  - Targeted electrode placement in future studies",
        "  - Hypothesis generation about neural mechanisms",
        "  - Clinical interpretation of model predictions",
        "",
        "="*70,
    ])

    report_text = "\\n".join(report_lines)
    print(report_text)

    # Save to file
    with open(output_path / f"{model_name}_xai_summary.txt", "w") as f:
        f.write(report_text)

    return report_text

# Example: Generate summary for tri-modal results
# if 'trimodal_results' in dir():
#     generate_xai_summary(trimodal_results, EEG_CHANNEL_NAMES, "trimodal",
#                          config.output_dir / "xai_analysis")

print("Summary generation function defined - call generate_xai_summary() after analysis")'''
    }
]

# Convert source to list format for notebook
for cell in xai_cells:
    source = cell['source']
    cell['source'] = [line + '\n' for line in source.split('\n')[:-1]] + [source.split('\n')[-1]]

# Find position to insert (before last cell)
insert_position = len(nb['cells']) - 1

# Insert cells
for i, cell in enumerate(xai_cells):
    nb['cells'].insert(insert_position + i, cell)

# Save notebook
with open('CrossModal_V4_final_0.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f'Added {len(xai_cells)} XAI cells')
print(f'Total cells after: {len(nb["cells"])}')
