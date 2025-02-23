def format_fairness_data(raw_data: dict):
    """
    Processes raw fairness data (e.g., from a backend API) into a format for visualization.
    Expects raw_data to contain 'group_accuracies' and 'overall_accuracy'.
    Returns a dictionary with lists of groups and corresponding accuracies.
    """
    if "group_accuracies" not in raw_data:
        return {}
    groups = list(raw_data["group_accuracies"].keys())
    accuracies = list(raw_data["group_accuracies"].values())
    formatted = {
        "groups": groups,
        "accuracies": accuracies,
        "overall_accuracy": raw_data.get("overall_accuracy")
    }
    return formatted
