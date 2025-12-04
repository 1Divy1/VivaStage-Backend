def calculate_max_reels(duration_seconds: int) -> int:
    """
    Calculate the maximum number of reels based on video duration.

    Args:
        duration_seconds: Video duration in seconds

    Returns:
        Maximum number of reels:
        - 0-10 min (0-600s): 6 reels max
        - 10-20 min (601-1200s): 10 reels max
        - 20-30 min (1201-1800s): 12 reels max
        - 30-60 min (1801-3600s): 15 reels max
        - 60+ min (3601+s): 18 reels max
    """
    if duration_seconds <= 600:  # 0-10 minutes
        return 6
    elif duration_seconds <= 1200:  # 10-20 minutes
        return 10
    elif duration_seconds <= 1800:  # 20-30 minutes
        return 12
    elif duration_seconds <= 3600:  # 30-60 minutes
        return 15
    else:  # 60+ minutes
        return 18


def format_duration(duration_seconds: int) -> str:
    """
    Format duration in seconds to human-readable format.

    Args:
        duration_seconds: Duration in seconds

    Returns:
        Formatted duration string (e.g., "1h 23m 45s", "23m 45s", "45s")
    """
    hours = duration_seconds // 3600
    minutes = (duration_seconds % 3600) // 60
    seconds = duration_seconds % 60

    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if seconds > 0 or not parts:  # Always show seconds if it's the only unit
        parts.append(f"{seconds}s")

    return " ".join(parts)