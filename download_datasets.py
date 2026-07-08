"""
Download and save actual HuggingFace datasets to local disk.
Modified to include Wikitext-2, MetaMathQA, and GSM8K.
"""

import os

os.environ["HF_HOME"] = os.environ.get('SCRATCH')

from datasets import load_dataset
import argparse




def download_generic_dataset(output_dir, repo_id, folder_name, num_samples=None, subset=None):
    """Generic helper to reduce redundancy for simple dataset downloads."""
    print("\n" + "="*80)
    print(f"Downloading {folder_name.upper()} Dataset")
    print("="*80)
    print(f"Source: {repo_id}")

    try:
        print("\n📥 Downloading from HuggingFace...")
        # Load dataset
        if subset:
            dataset = load_dataset(repo_id, subset)
        else:
            dataset = load_dataset(repo_id)

        print(f"✓ Dataset loaded")
        
        # Display and limit samples across all splits
        for split in dataset.keys():
            print(f"    {split}: {len(dataset[split])} examples")
            if num_samples:
                print(f"    ✂️  Limiting {split} to {num_samples} samples")
                dataset[split] = dataset[split].select(range(min(num_samples, len(dataset[split]))))

        # Save to disk
        save_path = os.path.join(output_dir, "datasets", folder_name)
        os.makedirs(save_path, exist_ok=True)

        print(f"\n💾 Saving to: {save_path}")
        dataset.save_to_disk(save_path)
        print(f"✓ {folder_name} dataset saved successfully!")
        return True

    except Exception as e:
        print(f"❌ Error downloading {folder_name}: {e}")
        return False


def download_openhermes(output_dir, num_samples=None):
    return download_generic_dataset(output_dir, "teknium/openhermes", "openhermes", num_samples)

def download_wizardlm(output_dir, num_samples=None):
    return download_generic_dataset(output_dir, "WizardLMTeam/WizardLM_evol_instruct_70k", "wizardlm", num_samples)

def download_code_feedback(output_dir, num_samples=None):
    return download_generic_dataset(output_dir, "m-a-p/CodeFeedback-Filtered-Instruction", "code-feedback", num_samples)

def download_openorca(output_dir, num_samples=None):
    return download_generic_dataset(output_dir, "Open-Orca/OpenOrca", "openorca", num_samples)

def download_alpaca(output_dir, num_samples=None):
    return download_generic_dataset(output_dir, "tatsu-lab/alpaca", "alpaca", num_samples)

def download_ai2_arc(output_dir, num_samples=None, subset="ARC-Challenge"):
    return download_generic_dataset(output_dir, "allenai/ai2_arc", f"ai2_arc_{subset.lower()}", num_samples, subset=subset)

def download_wikitext(output_dir, num_samples=None):
    return download_generic_dataset(output_dir, "wikitext", "wikitext-2-raw-v1", num_samples, subset="wikitext-2-raw-v1")

def download_metamath(output_dir, num_samples=None):
    return download_generic_dataset(output_dir, "meta-math/MetaMathQA", "metamathqa", num_samples)

def download_gsm8k(output_dir, num_samples=None):
    return download_generic_dataset(output_dir, "gsm8k", "gsm8k", num_samples, subset="main")


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get('SCRATCH', './data'),
        help="Output directory (defaults to $SCRATCH or ./data)"
    )
    
    available_choices = [
        "openhermes", "wizardlm", "code-feedback", "openorca", 
        "alpaca", "ai2_arc", "wikitext", "metamath", "gsm8k", "all"
    ]
    
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["openhermes", "wizardlm"],
        choices=available_choices,
        help="Which datasets to download"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Limit to first N samples per split (for testing)"
    )

    args = parser.parse_args()

    # Handle 'all' option
    if "all" in args.datasets:
        args.datasets = [d for d in available_choices if d != "all"]

    print("="*80)
    print("HuggingFace Dataset Downloader")
    print("="*80)
    print(f"\n📁 Output directory: {args.output_dir}")
    print(f"📦 Datasets to download: {', '.join(args.datasets)}")

    results = {}

    # Map dataset keys to functions
    dispatch = {
        "openhermes": download_openhermes,
        "wizardlm": download_wizardlm,
        "code-feedback": download_code_feedback,
        "openorca": download_openorca,
        "alpaca": download_alpaca,
        "ai2_arc": download_ai2_arc,
        "wikitext": download_wikitext,
        "metamath": download_metamath,
        "gsm8k": download_gsm8k
    }

    for ds_key in args.datasets:
        if ds_key in dispatch:
            results[ds_key] = dispatch[ds_key](args.output_dir, args.num_samples)

    # Summary
    print("\n" + "="*80)
    print("Download Summary")
    print("="*80)

    for dataset, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{dataset:15}: {status}")

    all_success = all(results.values())
    print("="*80)
    return 0 if all_success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())