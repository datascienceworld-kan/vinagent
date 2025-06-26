import re

class Utils:
    @staticmethod
    def look_like_markdown(text: str) -> bool:

        if not isinstance(text, str) or len(text) < 10:
            return False

        markdown_patterns = [
            r'^#+\s',
            r'^[*-+]\s',
            r'^\d+\.\s',
            r'^\>\s',
            r'\*\*.*\*\*',
            r'__.*__',
            r'\*.*\*',
            r'_.*_',
            r'`.*`',
            r'```',
            r'^---\s*$',
            r'^___\s*$',
            r'\|.*\|',
            r'\[.*\]\(.*\)'
        ]

        for pattern in markdown_patterns:
            if re.search(pattern, text, re.M):
                return True

        return False