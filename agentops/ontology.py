
def classify_log(raw: str) -> str:
    text = raw.lower()

    ios_keywords = ["provision", "codesign", "entitlement", "xcode"]
    android_keywords = ["gradle", "keystore", "manifest", "kotlin"]
    dep_keywords = ["dependency", "version conflict", "could not resolve"]
    network_keywords = ["timeout", "network", "dns", "connection"]
    test_keywords = ["assertion failed", "test failed", "unit test"]

    if any(k in text for k in ios_keywords):
        return "ios_build_error"
    if any(k in text for k in android_keywords):
        return "android_build_error"
    if any(k in text for k in dep_keywords):
        return "dependency_error"
    if any(k in text for k in network_keywords):
        return "network_error"
    if any(k in text for k in test_keywords):
        return "test_failure"

    return "unknown"
