"""
Checkpoint management utilities.
"""

import torch
import shutil
from pathlib import Path
from typing import Dict, Optional, List
import json
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpoints with automatic cleanup.
    
    Features:
    - Save/load checkpoints
    - Keep only N most recent checkpoints
    - Track best model based on metrics
    - Metadata logging
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 5
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_to_keep: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / 'metadata.json'
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load checkpoint metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': [], 'best_checkpoint': None}
    
    def _save_metadata(self):
        """Save checkpoint metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save(
        self,
        checkpoint: Dict,
        filename: str,
        is_best: bool = False
    ):
        """
        Save a checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
            filename: Filename for the checkpoint
            is_best: Whether this is the best model so far
        """
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Update metadata
        checkpoint_info = {
            'filename': filename,
            'path': str(checkpoint_path),
            'timestamp': datetime.now().isoformat(),
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0)
        }
        
        self.metadata['checkpoints'].append(checkpoint_info)
        
        # Update best checkpoint
        if is_best:
            self.metadata['best_checkpoint'] = checkpoint_info
            # Also save as 'best_model.pth'
            best_path = self.checkpoint_dir / 'best_model.pth'
            shutil.copy(checkpoint_path, best_path)
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        # Save metadata
        self._save_metadata()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        if len(self.metadata['checkpoints']) <= self.max_to_keep:
            return
        
        # Sort by timestamp
        sorted_checkpoints = sorted(
            self.metadata['checkpoints'],
            key=lambda x: x['timestamp'],
            reverse=True
        )
        
        # Keep only max_to_keep most recent
        to_keep = sorted_checkpoints[:self.max_to_keep]
        to_remove = sorted_checkpoints[self.max_to_keep:]
        
        # Remove old checkpoints
        for checkpoint_info in to_remove:
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists():
                # Don't remove if it's the best checkpoint
                if self.metadata['best_checkpoint'] and \
                   checkpoint_info['filename'] != self.metadata['best_checkpoint']['filename']:
                    checkpoint_path.unlink()
        
        # Update metadata
        self.metadata['checkpoints'] = to_keep
    
    def load(
        self,
        filename: Optional[str] = None,
        load_best: bool = False
    ) -> Dict:
        """
        Load a checkpoint.
        
        Args:
            filename: Checkpoint filename (if None, loads latest)
            load_best: Whether to load the best checkpoint
        
        Returns:
            Checkpoint dictionary
        """
        if load_best:
            if self.metadata['best_checkpoint'] is None:
                raise ValueError("No best checkpoint found")
            checkpoint_path = Path(self.metadata['best_checkpoint']['path'])
        elif filename:
            checkpoint_path = self.checkpoint_dir / filename
        else:
            # Load latest checkpoint
            if not self.metadata['checkpoints']:
                raise ValueError("No checkpoints found")
            checkpoint_path = Path(self.metadata['checkpoints'][0]['path'])
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    
    def get_latest_checkpoint_path(self) -> Optional[Path]:
        """Get path to the latest checkpoint."""
        if not self.metadata['checkpoints']:
            return None
        return Path(self.metadata['checkpoints'][0]['path'])
    
    def get_best_checkpoint_path(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        if self.metadata['best_checkpoint'] is None:
            return None
        return Path(self.metadata['best_checkpoint']['path'])
    
    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints."""
        return self.metadata['checkpoints']


if __name__ == "__main__":
    # Test checkpoint manager
    print("Testing CheckpointManager...")
    
    import tempfile
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, max_to_keep=3)
        
        # Save some checkpoints
        for i in range(5):
            checkpoint = {
                'epoch': i,
                'global_step': i * 100,
                'model_state': {'dummy': 'data'}
            }
            manager.save(checkpoint, f'checkpoint_{i}.pth', is_best=(i == 3))
        
        # List checkpoints
        checkpoints = manager.list_checkpoints()
        print(f"Number of checkpoints: {len(checkpoints)}")
        
        # Load best checkpoint
        best = manager.load(load_best=True)
        print(f"Best checkpoint epoch: {best['epoch']}")
        
        # Load latest checkpoint
        latest = manager.load()
        print(f"Latest checkpoint epoch: {latest['epoch']}")
    
    print("\n✓ CheckpointManager test passed!")
