class ConsoleNotifier:
    def notify(self, analysis):
        print("\n=== AgentOps Report ===")
        print(f"Main Category : {analysis.main_category}")
        print(f"Sub Category  : {analysis.category}")
        print(f"Root Cause    : {analysis.root_cause}")
        print(f"Explanation   : {analysis.explanation}")
        print(f"Suggestion    : {analysis.suggestion}")
        print(f"Confidence    : {analysis.confidence}")
        if analysis.affected_files:
            print(f"Affected Files: {', '.join(analysis.affected_files)}")
        if analysis.affected_classes:
            print(f"Affected Class: {', '.join(analysis.affected_classes)}")
        if analysis.conflict_type:
            print(f"Conflict Type : {analysis.conflict_type}")
        if analysis.resolution_strategy:
            print(f"Resolution    : {analysis.resolution_strategy}")
        if analysis.jira_task_id:
            print(f"Jira Task     : {analysis.jira_task_id}")
        print("=======================\n")
