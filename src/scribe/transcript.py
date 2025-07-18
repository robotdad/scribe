"""Transcript-specific processing and optimization."""

import re
from typing import Dict, Any, List


class TranscriptProcessor:
    """Handles transcript-specific processing and optimization."""
    
    def optimize(self, markdown: str) -> str:
        """
        Optimize markdown for transcript readability.
        
        Args:
            markdown: Raw markdown content
        
        Returns:
            Optimized transcript content
        """
        lines = markdown.split('\n')
        optimized_lines = []
        
        for line in lines:
            # Detect and format speaker lines
            speaker_match = re.match(r'^(\[.*?\])\s*(.+?):', line)
            if speaker_match:
                timestamp, speaker = speaker_match.groups()
                line = f"\n### {timestamp} {speaker}:\n"
            
            # Clean up common artifacts
            line = re.sub(r'\s+', ' ', line)  # Multiple spaces
            line = re.sub(r'^\s*[-•]\s*', '- ', line)  # Bullet points
            
            optimized_lines.append(line)
        
        return '\n'.join(optimized_lines)
    
    def extract_metadata(self, markdown: str) -> Dict[str, Any]:
        """
        Extract structured information from transcript.
        
        Args:
            markdown: Transcript content
        
        Returns:
            Extracted metadata
        """
        speakers = []
        timestamps = []
        
        # Extract speakers and timestamps
        speaker_pattern = re.compile(r'\[(.*?)\]\s*(.+?):')
        for match in speaker_pattern.finditer(markdown):
            timestamp, speaker = match.groups()
            if speaker not in speakers:
                speakers.append(speaker)
            timestamps.append({
                "time": timestamp,
                "speaker": speaker,
                "position": match.start()
            })
        
        return {
            "speakers": speakers,
            "speaker_count": len(speakers),
            "timestamps": timestamps,
            "total_exchanges": len(timestamps)
        }