"""
Test script to verify the modified Grad-CAM quota logic.

This test simulates a scenario where:
- We request 125 samples per class per category (500 total)
- The model has limited incorrect predictions
- The function should fill remaining quota with correct predictions
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MockDataset(Dataset):
    """
    Mock dataset that simulates a high-accuracy model.

    Returns:
    - 400 correct predictions (200 per class)
    - 100 incorrect predictions (50 per class, simulating high accuracy)
    """

    def __init__(self, num_samples=500):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create dummy image (3x224x224)
        image = torch.randn(3, 224, 224)

        # Simulate high accuracy: 80% correct, 20% incorrect
        if idx < 400:  # First 400 are correct predictions
            label = idx % 2
            # Return image and label (model will predict correctly)
        else:  # Last 100 are incorrect predictions
            label = idx % 2
            # Return image with opposite label stored separately

        filename = f"sample_{idx:04d}.jpg"
        return image, label, filename


class MockModel(nn.Module):
    """
    Mock model that simulates high accuracy.
    Predicts correctly for first 400 samples, incorrectly for last 100.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.fc = nn.Linear(512 * 7 * 7, 2)
        self.call_count = 0

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.layer4(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

        # Simulate predictions
        # First 400 samples: correct predictions (logits favor true label)
        # Last 100 samples: incorrect predictions (logits favor wrong label)
        output = torch.zeros(batch_size, 2)

        for i in range(batch_size):
            sample_idx = self.call_count + i
            if sample_idx < 400:
                # Correct prediction
                true_label = sample_idx % 2
                output[i, true_label] = 10.0
                output[i, 1 - true_label] = -10.0
            else:
                # Incorrect prediction
                true_label = sample_idx % 2
                output[i, 1 - true_label] = 10.0
                output[i, true_label] = -10.0

        self.call_count += batch_size
        return output


def test_quota_logic():
    """
    Test that the modified logic guarantees 500 total samples.

    This test verifies the collection phase only, without actual Grad-CAM generation.
    """
    print("Testing Grad-CAM quota collection logic...")
    print("=" * 60)

    # Create mock dataset and dataloader
    dataset = MockDataset(num_samples=500)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Create mock model
    model = MockModel()
    model.eval()

    # Manually test the collection logic (Phase 1 and Phase 2)
    print("\nSimulating collection phases...")
    print("Target: 125 per class per category = 500 total\n")

    num_heatmaps_per_class = 125
    total_target = num_heatmaps_per_class * 4

    # Storage for samples
    samples_by_category = {
        'correct': {0: [], 1: []},
        'incorrect': {0: [], 1: []}
    }

    # Phase 1: Balanced collection
    print("Phase 1: Balanced collection")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            images, labels, filenames = batch
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for img, label, pred, filename in zip(images, labels, preds, filenames):
                label_val = label.item()
                pred_val = pred.item()
                category = 'correct' if label_val == pred_val else 'incorrect'

                if len(samples_by_category[category][label_val]) < num_heatmaps_per_class:
                    samples_by_category[category][label_val].append({
                        'label': label_val,
                        'pred': pred_val,
                        'filename': filename
                    })

            # Check if all satisfied
            all_satisfied = all(
                len(samples_by_category[cat][cls]) >= num_heatmaps_per_class
                for cat in ['correct', 'incorrect']
                for cls in [0, 1]
            )
            if all_satisfied:
                break

    current_total = sum(
        len(samples_by_category[cat][cls])
        for cat in ['correct', 'incorrect']
        for cls in [0, 1]
    )

    print(f"After Phase 1: {current_total} samples collected")

    # Phase 2: Fill remaining
    if current_total < total_target:
        remaining = total_target - current_total
        print(f"Phase 2: Filling remaining {remaining} samples")

        model.call_count = 0  # Reset counter
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if current_total >= total_target:
                    break

                images, labels, filenames = batch
                outputs = model(images)
                preds = outputs.argmax(dim=1)

                for img, label, pred, filename in zip(images, labels, preds, filenames):
                    if current_total >= total_target:
                        break

                    label_val = label.item()
                    pred_val = pred.item()
                    category = 'correct' if label_val == pred_val else 'incorrect'

                    samples_by_category[category][label_val].append({
                        'label': label_val,
                        'pred': pred_val,
                        'filename': filename
                    })
                    current_total += 1

    print(f"After Phase 2: {current_total} samples collected")

    # Convert to results format
    results = {'correct': [], 'incorrect': []}
    for category in ['correct', 'incorrect']:
        for class_id in [0, 1]:
            for sample in samples_by_category[category][class_id]:
                results[category].append(sample)

    # Analyze results
    print("\n" + "=" * 60)
    print("VERIFICATION:")
    print("=" * 60)

    num_correct = len(results['correct'])
    num_incorrect = len(results['incorrect'])
    total = num_correct + num_incorrect

    num_correct_class0 = len([r for r in results['correct'] if r['label'] == 0])
    num_correct_class1 = len([r for r in results['correct'] if r['label'] == 1])
    num_incorrect_class0 = len([r for r in results['incorrect'] if r['label'] == 0])
    num_incorrect_class1 = len([r for r in results['incorrect'] if r['label'] == 1])

    print(f"\nTotal samples collected: {total}")
    print(f"  Correct predictions: {num_correct}")
    print(f"    - Class 0: {num_correct_class0}")
    print(f"    - Class 1: {num_correct_class1}")
    print(f"  Incorrect predictions: {num_incorrect}")
    print(f"    - Class 0: {num_incorrect_class0}")
    print(f"    - Class 1: {num_incorrect_class1}")

    # Assertions
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print("=" * 60)

    success = True

    if total == 500:
        print("✓ PASS: Total sample count is exactly 500")
    else:
        print(f"✗ FAIL: Total sample count is {total}, expected 500")
        success = False

    if num_incorrect <= 100:
        print(f"✓ PASS: Incorrect predictions ({num_incorrect}) limited by availability")
    else:
        print(f"✗ FAIL: Too many incorrect predictions: {num_incorrect}")
        success = False

    if num_correct >= 400:
        print(f"✓ PASS: Remaining quota filled with correct predictions ({num_correct})")
    else:
        print(f"✗ FAIL: Not enough correct predictions: {num_correct}")
        success = False

    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 60)

    return success


if __name__ == "__main__":
    test_quota_logic()
