# src/agentops/cli.py

import os
import requests
from dotenv import load_dotenv

from .log_models import CiLog
from .analyzer import SimpleAnalyzer, OllamaAnalyzer
from .notifier import ConsoleNotifier
from .feedback_service import _BUILD_STATUS_MAP, _to_build_status

load_dotenv()

N8N_WEBHOOK_URL        = os.getenv("N8N_WEBHOOK_URL", "")
BITBUCKET_HOST         = os.getenv("BITBUCKET_HOST", "")
BITBUCKET_PR_WORKSPACE = os.getenv("BITBUCKET_PR_WORKSPACE", "")
BITBUCKET_PR_REPO      = os.getenv("BITBUCKET_PR_REPO", "")


def _extract_pr_id(run_id: str) -> str:
    """Jenkins BUILD_TAG'inden PR numarasını çıkarır. Örn: jenkins-example-ios-PR-42-123 → '42'"""
    import re
    m = re.search(r'PR-(\d+)', run_id)
    return m.group(1) if m else ""


# --- Örnek loglar ---

LOGS = {
    "pbxproj_conflict": {
        "platform": "ios",
        "pipeline_name": "example-ios-pipeline",
        "run_id": 1001,
        "branch": "stage/MOB-12345-payment-screen",
        "target_branch": "develop",
        "pr_id": "9200",
        "commit_sha": "a1b2c3d4",
        "raw_log": """
[Pipeline] stage: Build
+ xcodebuild -workspace ExampleApp.xcworkspace -scheme ExampleApp-Release -configuration Release

error: Build input file cannot be found:
'/Users/jenkins/workspace/example-ios/Pods/FirebaseCore/Sources/FIRApp.m'

CONFLICT (content): Merge conflict in ExampleApp.xcodeproj/project.pbxproj
Auto-merging ExampleApp.xcodeproj/project.pbxproj
Automatic merge failed; fix conflicts and then commit the result.

<<<<<<< HEAD
        A1B2C3D4E5F6 /* PaymentViewController.swift in Sources */ = {
            isa = PBXBuildFile;
            fileRef = A1B2C3D4E5F6;
        };
=======
        F6E5D4C3B2A1 /* PaymentViewController.swift in Sources */ = {
            isa = PBXBuildFile;
            fileRef = F6E5D4C3B2A1;
        };
>>>>>>> feature/payment-screen

Build failed with exit code 65
""",
    },

    "cocoapods_error": {
        "platform": "ios",
        "pipeline_name": "example-ios-pipeline",
        "run_id": 1002,
        "branch": "feature/firebase-update",
        "commit_sha": "b2c3d4e5",
        "raw_log": """
[Pipeline] stage: Pod Install
+ pod install --repo-update

Analyzing dependencies
[!] CocoaPods could not find compatible versions for pod "Firebase/Analytics":
  In Podfile:
    Firebase/Analytics (= 10.15.0)

Specs satisfying the `Firebase/Analytics (= 10.15.0)` dependency were found,
but they required a higher minimum deployment target.

None of your spec sources contain a spec satisfying the dependencies:
`Firebase/Analytics (= 10.15.0)`

You have either:
 * out-of-date source repos which you can update with `pod repo update` or with `pod install --repo-update`.
 * changed the constraints of dependency `Firebase/Analytics` inside your development pod `ExampleApp`.

pod install failed with exit code 1
""",
    },

    "signing_error": {
        "platform": "ios",
        "pipeline_name": "example-ios-pipeline",
        "run_id": 1003,
        "branch": "release/2.4.0",
        "commit_sha": "c3d4e5f6",
        "raw_log": """
[Pipeline] stage: Archive
+ xcodebuild archive -workspace ExampleApp.xcworkspace -scheme ExampleApp-Release

error: No signing certificate "iOS Distribution" found:
No 'iOS Distribution' signing certificate matching team ID 'XYZ123ABC'
with a non-expired certificate was found.

CodeSign error: code signing is required for product type
'Application' in SDK 'iOS 16.4'

Check your Provisioning Profile and Certificate settings.

Build failed with exit code 65
""",
    },

    "spm_dns_exit74": {
        "platform": "ios",
        "pipeline_name": "EXAMPLE-IOS/deployToDistro",
        "run_id": 1846,
        "branch": "develop",
        "commit_sha": "b9bf5fde3daa5254a7e0259fd59ff7dafadf5cba",
        "raw_log": r"""
Started by user Selen Naz Ercan

[Pipeline] Start of Pipeline
[Pipeline] node
Running on iosagent-1
 in /Users/macstudio/jenkins/jenkins_slave/workspace/EXAMPLE-IOS/deployToDistro
[Pipeline] {
[Pipeline] withEnv
[Pipeline] {
[Pipeline] stage
[Pipeline] { (Unlock Keychain)
[Pipeline] sh
+ security unlock-keychain -p '<REDACTED>'
[Pipeline] }
[Pipeline] // stage
[Pipeline] stage
[Pipeline] { (Git Clone)
[Pipeline] checkout
The recommended git tool is: NONE
using credential BitbucketoAuthConsumer
Cloning the remote Git repository
Using shallow clone with depth 1
Avoid fetching tags
Cloning repository ssh://git@bitbucket.example.com/mob-example/example-ios.git
 > git init /Users/macstudio/jenkins/jenkins_slave/workspace/EXAMPLE-IOS/deployToDistro # timeout=10
Fetching upstream changes from ssh://git@bitbucket.example.com/mob-example/example-ios.git
 > git --version # timeout=10
 > git --version # 'git version 2.50.1 (Apple Git-155)'
using GIT_ASKPASS to set credentials Bitbucket oAuth Consumer
 > git fetch --no-tags --force --progress --depth=1 -- ssh://git@bitbucket.example.com/mob-example/example-ios.git +refs/heads/*:refs/remotes/origin/* # timeout=30
Avoid second fetch
Checking out Revision b9bf5fde3daa5254a7e0259fd59ff7dafadf5cba (origin/develop)
Commit message: "Pull request #9115: Stage/PROJ-23576 password rules logging"
First time build. Skipping changelog.
[Pipeline] }
[Pipeline] // stage
[Pipeline] stage
[Pipeline] { (Pod Install)
[Pipeline] script
[Pipeline] {
[Pipeline] fileExists
[Pipeline] echo
Podfile not found, skipping pod install.
[Pipeline] }
[Pipeline] // script
[Pipeline] }
[Pipeline] // stage
[Pipeline] stage
[Pipeline] { (Resolve SPM Packages)
[Pipeline] fileExists
[Pipeline] script
[Pipeline] {
[Pipeline] fileExists
[Pipeline] sh
+ xcodebuild -resolvePackageDependencies -project ExampleApp.xcodeproj
Command line invocation:
    /Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -resolvePackageDependencies -project ExampleApp.xcodeproj

resolved source packages:
[Pipeline] }
[Pipeline] // script
[Pipeline] }
[Pipeline] // stage
[Pipeline] stage
[Pipeline] { (Fastlane)
[Pipeline] sh
 > git config remote.origin.url ssh://git@bitbucket.example.com/mob-example/example-ios.git # timeout=10
 > git config --add remote.origin.fetch +refs/heads/*:refs/remotes/origin/* # timeout=10
 > git rev-parse origin/develop^{commit} # timeout=10
 > git config core.sparsecheckout # timeout=10
 > git checkout -f b9bf5fde3daa5254a7e0259fd59ff7dafadf5cba # timeout=10
 > git rev-list --no-walk abfc8409e63a618ca49052aa1b1bb489319031c6 # timeout=10
+ fastlane deployTestToNewDistro branchName:develop distro_token:<REDACTED_TOKEN> environment:Test app_id:1
[17:21:50]: fastlane detected a Gemfile in the current directory
[17:21:50]: However, it seems like you didn't use `bundle exec`
[17:21:50]: To launch fastlane faster, please use
[17:21:50]:
[17:21:50]: $ bundle exec fastlane deployTestToNewDistro branchName:develop distro_token:<REDACTED_TOKEN> environment:Test app_id:1
[17:21:50]:
[17:21:50]: Get started using a Gemfile for fastlane https://docs.fastlane.tools/getting-started/ios/setup/#use-a-gemfile
[17:21:51]: ------------------------------
[17:21:51]: --- Step: default_platform ---
[17:21:51]: ------------------------------
[17:21:51]: Driving the lane 'ios deployTestToNewDistro'
[17:21:51]: -----------------------------
[17:21:51]: --- Step: ci_build_number ---
[17:21:51]: -----------------------------
[17:21:51]: ---------------------------------------------
[17:21:51]: --- Step: increment_build_number_in_plist ---
[17:21:51]: ---------------------------------------------
[17:21:51]: Resolving Swift Package Manager dependencies...
[17:21:51]: $ xcodebuild -resolvePackageDependencies -scheme ExampleApp\ Test -project ExampleApp.xcodeproj -configuration Test
[17:21:51]: Command line invocation:
[17:21:51]:     /Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -resolvePackageDependencies -scheme "ExampleApp Test" -project ExampleApp.xcodeproj -configuration Test
[17:21:52]: resolved source packages:
[17:21:52]: $ xcodebuild -showBuildSettings -scheme ExampleApp\ Test -project ExampleApp.xcodeproj -configuration Test 2>&1
[17:21:54]: will continue and update the info plist key. this will replace the existing value.
[17:21:54]: -------------------------------------------
[17:21:54]: --- Step: get_version_number_from_plist ---
[17:21:54]: -------------------------------------------
[17:21:54]: Resolving Swift Package Manager dependencies...
[17:21:54]: $ xcodebuild -resolvePackageDependencies -scheme ExampleApp\ Test -project ExampleApp.xcodeproj -configuration Test
[17:21:55]: Command line invocation:
[17:21:55]:     /Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -resolvePackageDependencies -scheme "ExampleApp Test" -project ExampleApp.xcodeproj -configuration Test
[17:21:56]: resolved source packages:
[17:21:56]: $ xcodebuild -showBuildSettings -scheme ExampleApp\ Test -project ExampleApp.xcodeproj -configuration Test 2>&1
[17:21:58]: version will originate from xcodeproj
[17:21:58]: Resolving Swift Package Manager dependencies...
[17:21:58]: $ xcodebuild -resolvePackageDependencies -scheme ExampleApp\ Test -project ExampleApp.xcodeproj -configuration Test
[17:21:59]: Command line invocation:
[17:21:59]:     /Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -resolvePackageDependencies -scheme "ExampleApp Test" -project ExampleApp.xcodeproj -configuration Test
[17:22:00]: resolved source packages:
[17:22:00]: $ xcodebuild -showBuildSettings -scheme ExampleApp\ Test -project ExampleApp.xcodeproj -configuration Test 2>&1
[17:22:02]: ------------------------------
[17:22:02]: --- Step: get_build_number ---
[17:22:02]: ------------------------------
[17:22:02]: $ cd /Users/macstudio/jenkins/jenkins_slave/workspace/EXAMPLE-IOS/deployToDistro && agvtool what-version -terse
[17:22:02]: 10002
[17:22:02]: -----------------------------
[17:22:02]: --- Step: ci_build_number ---
[17:22:02]: -----------------------------
[17:22:02]: ci_build_number 1846
[17:22:02]: -----------------------------------------------------------
[17:22:02]: --- Step: cd .. && bash generate_swifttlint_filelist.sh ---
[17:22:02]: -----------------------------------------------------------
[17:22:02]: $ cd .. && bash generate_swifttlint_filelist.sh
[17:22:02]: Skipping DataProvider - directory not found
[17:22:02]: Skipping Utilities - directory not found
[17:22:02]: Skipping PayDataProvider - directory not found
[17:22:02]: -----------------------------
[17:22:02]: --- Step: ci_build_number ---
[17:22:02]: -----------------------------
[17:22:02]: -----------------
[17:22:02]: --- Step: gym ---
[17:22:02]: -----------------
[17:22:02]: Resolving Swift Package Manager dependencies...
[17:22:02]: $ xcodebuild -resolvePackageDependencies -workspace ./ExampleApp.xcworkspace -scheme ExampleApp\ Test
[17:22:03]: Command line invocation:
[17:22:03]:     /Applications/Xcode.app/Contents/Developer/usr/bin/xcodebuild -resolvePackageDependencies -workspace ./ExampleApp.xcworkspace -scheme "ExampleApp Test"
[17:22:04]: 2026-04-16 17:22:04.680 xcodebuild[17021:20460108] [MT] IDERunDestination: Supported platforms for the buildables in the current scheme is empty.
[17:22:04]: Resolve Package Graph
[17:22:04]: 2026-04-16 17:22:04.840 xcodebuild[17021:20460108] [MT] IDERunDestination: Supported platforms for the buildables in the current scheme is empty.
[17:22:27]: Updating from https://github.com/google/grpc-binary.git
[17:26:16]: Updating from https://github.com/ephread/Instructions.git
[17:26:16]: Updating from https://github.com/onevcat/Kingfisher.git
[17:26:16]: Updating from https://github.com/krzyzanowskim/CryptoSwift
[17:26:16]: Updating from https://github.com/google/GoogleDataTransport.git
[17:26:16]: Updating from https://github.com/hyperoslo/Lightbox
[17:26:16]: Updating from https://github.com/teodorpatras/EasyTipView
[17:26:16]: Updating from https://github.com/mobillium/MobilliumDateFormatter
[17:26:16]: Updating from https://github.com/google/abseil-cpp-binary.git
[17:26:16]: Updating from https://github.com/mobillium/MobilliumBuilders
[17:26:16]: Updating from https://github.com/adjust/ios_sdk
[17:26:16]: Updating from https://github.com/firebase/firebase-ios-sdk.git
[17:26:16]: Updating from https://github.com/dataroid/dataroid-sdk-ios
[17:26:16]: Updating from https://github.com/airbnb/lottie-spm.git
[17:26:16]: Updating from https://github.com/firebase/nanopb.git
[17:26:16]: Updating from https://gitea.example.com/internal/ArkTCKKSPM.git
[17:26:16]: Updating from https://github.com/kasketis/netfox
[17:26:16]: Updating from https://github.com/hackiftekhar/IQKeyboardManager.git
[17:26:16]: Updating from https://github.com/google/GoogleUtilities.git
[17:26:16]: Updating from https://bitbucket.example.com/scm/moba/examplesdk-ios.git
[17:26:16]: Updating from https://gitea.example.com/internal/ArkFaceSPM.git
[17:26:16]: Updating from https://github.com/RedMadRobot/input-mask-ios
[17:26:16]: Updating from https://github.com/roberthein/TinyConstraints
[17:26:16]: Updating from https://github.com/mobillium/MBLDeviceModelHelper
[17:26:16]: Updating from https://github.com/Netvent/storyly-ios
[17:26:16]: Updating from https://github.com/kishikawakatsumi/KeychainAccess
[17:26:16]: Updating from https://bitbucket.example.com/scm/moba/exampleuikit-ios.git
[17:26:16]: Updating from https://github.com/mobillium/MobilliumUserDefaults
[17:26:16]: Updating from https://github.com/huri000/SwiftEntryKit
[17:26:16]: Updating from https://gitea.example.com/internal/ArkNFCSPM
[17:26:16]: Updating from https://github.com/WenchaoD/FSCalendar.git
[17:26:16]: Updating from https://github.com/firebase/leveldb.git
[17:26:16]: Updating from https://github.com/apple/swift-protobuf.git
[17:26:16]: Updating from https://github.com/google/promises.git
[17:26:16]: Updating from https://gitea.example.com/internal/ArkTCNCSPM.git
[17:26:16]: Updating from https://github.com/google/gtm-session-fetcher.git
[17:26:16]: Updating from https://github.com/google/app-check.git
[17:26:16]: Updating from https://github.com/hmlongco/Factory
[17:26:16]: Updating from https://gitea.example.com/internal/ArkPspSPM.git
[17:26:16]: Updating from https://gitea.example.com/internal/ArkVideoCall.git
[17:26:16]: Updating from https://github.com/google/GoogleAppMeasurement.git
[17:26:16]: Updating from https://github.com/Alamofire/Alamofire.git
[17:26:16]: Updating from https://github.com/google/interop-ios-for-google-sdks.git
[17:26:16]: Updating from https://github.com/skywinder/ActionSheetPicker-3.0
[17:26:16]: Updating from https://github.com/onevcat/Kingfisher
[17:26:16]: Updating from https://github.com/auth0/JWTDecode.swift
[17:26:16]: Updating from https://github.com/Alamofire/Alamofire
[17:26:16]: Updating from https://github.com/ephread/Instructions
[17:26:16]: Couldn’t fetch updates from remote repositories:
[17:26:16]:     fatal: unable to access 'https://github.com/krzyzanowskim/CryptoSwift/': Could not resolve host: github.com
[17:26:16]: xcodebuild: error: Could not resolve package dependencies:
[17:26:16]:   Couldn’t fetch updates from remote repositories:
[17:26:16]: Exit status: 74
[17:26:16]: Error: Exit status: 74
[17:26:16]: Called from Fastfile at line 89
[17:26:16]: Exit status: 74
[17:26:16]: fastlane finished with errors
[!] Exit status: 74
[Pipeline] }
[Pipeline] // stage
[Pipeline] stage
[Pipeline] { (DeployToTauto)
Stage "DeployToTauto" skipped due to earlier failure(s)
[Pipeline] getContext
[Pipeline] }
[Pipeline] // stage
[Pipeline] stage
[Pipeline] { (Declarative: Post Actions)
[Pipeline] echo
Cleaning up workspace...
[Pipeline] cleanWs
[WS-CLEANUP] Deleting project workspace...
[WS-CLEANUP] Deferred wipeout is used...
[WS-CLEANUP] done
[Pipeline] script
[Pipeline] {
[Pipeline] echo
Cleaning up iOS build files...
[Pipeline] }
[Pipeline] // script
[Pipeline] }
[Pipeline] // stage
[Pipeline] }
[Pipeline] // withEnv
[Pipeline] }
[Pipeline] // node
[Pipeline] End of Pipeline
ERROR: script returned exit code 1
Finished: FAILURE
""",
    },
}


def run_demo(log_key: str = "pbxproj_conflict"):
    """
    Seçilen örnek log ile pipeline'ı çalıştırır.

    log_key seçenekleri:
        - pbxproj_conflict  (merge conflict)
        - cocoapods_error   (dependency)
        - signing_error     (signing)
        - spm_dns_exit74    (spm dns/network)
    """
    if log_key not in LOGS:
        print(f"❌ Geçersiz log_key: {log_key}")
        print(f"   Seçenekler: {list(LOGS.keys())}")
        return

    log_data = LOGS[log_key]
    log = CiLog(
        platform=log_data["platform"],
        pipeline_name=log_data["pipeline_name"],
        run_id=log_data["run_id"],
        status="failure",
        raw_log=log_data["raw_log"],
        branch=log_data["branch"],
        commit_sha=log_data["commit_sha"],
        target_branch=log_data.get("target_branch", ""),
        pr_id=log_data.get("pr_id", ""),
    )

    print(f"\n{'='*60}")
    print(f"🚀 Demo: {log_key}")
    print(f"{'='*60}")

    analyzer = OllamaAnalyzer()
    notifier = ConsoleNotifier()

    analysis = analyzer.analyze(log)
    notifier.notify(analysis, log=log)


def run_analyze(run_id: str, platform: str, app: str, environment: str,
                reason: str, log_file: str) -> None:
    """
    Jenkins'ten çağrılır. Log dosyasını okur, analiz eder, n8n'e gönderir.

    Kullanım:
        python -m agentops.cli analyze \
          --run-id jenkins-example-ios-PR-42-123 \
          --platform iOS \
          --app ExampleApp \
          --environment staging \
          --reason "Stage failed: Git Clone" \
          --log-file /tmp/agentops_raw_log.txt
    """
    with open(log_file, "r", errors="replace") as f:
        raw_log = f.read()

    log = CiLog(
        platform=platform.lower(),
        pipeline_name="jenkins",
        run_id=run_id,
        status="failure",
        raw_log=raw_log,
        branch="",
        commit_sha="unknown",
    )

    analyzer = OllamaAnalyzer()
    notifier = ConsoleNotifier()

    analysis = analyzer.analyze(log)
    notifier.notify(analysis, log=log)

    build_status = _to_build_status(analysis.main_category)
    pr_id = _extract_pr_id(run_id)

    n8n_payload = {
        "build_status" : build_status,
        "run_id"       : run_id,
        "reason"       : reason,
        "platform"     : platform,
        "app"          : app,
        "environment"  : environment,
        "analysis"     : {
            "main_category"  : analysis.main_category,
            "category"       : analysis.category,
            "root_cause"     : analysis.root_cause,
            "explanation"    : analysis.explanation,
            "suggestion"     : analysis.suggestion,
            "confidence"     : analysis.confidence,
            "affected_files" : analysis.affected_files,
            "conflict_type"  : analysis.conflict_type or "",
            "jira_task_id"   : analysis.jira_task_id or "",
        },
        "pr": {
            "id"             : int(pr_id) if pr_id else "",
            "workspace"      : BITBUCKET_PR_WORKSPACE,
            "repo"           : BITBUCKET_PR_REPO,
            "bitbucket_host" : BITBUCKET_HOST,
        },
    }

    if N8N_WEBHOOK_URL:
        try:
            resp = requests.post(N8N_WEBHOOK_URL, json=n8n_payload, timeout=10)
            print(f"n8n → {resp.status_code}")
        except Exception as e:
            print(f"n8n gönderimi başarısız: {e}")
    else:
        print("N8N_WEBHOOK_URL tanımlı değil, gönderim atlandı.")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AgentOps CI Analyzer")
    subparsers = parser.add_subparsers(dest="command", help="Komut seçin")

    # analyze — Jenkins entegrasyonu
    analyze_parser = subparsers.add_parser("analyze", help="Jenkins log analiz et ve n8n'e gönder")
    analyze_parser.add_argument("--run-id",      required=True, help="Jenkins BUILD_TAG")
    analyze_parser.add_argument("--platform",    required=True, help="iOS | android")
    analyze_parser.add_argument("--app",         required=True, help="Uygulama adı")
    analyze_parser.add_argument("--environment", required=True, help="staging | production")
    analyze_parser.add_argument("--reason",      default="",    help="Stage failed: <stage>")
    analyze_parser.add_argument("--log-file",    required=True, help="Full log dosya yolu")

    # demo
    demo_parser = subparsers.add_parser("demo", help="Örnek log ile analiz çalıştır")
    demo_parser.add_argument(
        "log_key",
        nargs="?",
        default="pbxproj_conflict",
        choices=list(LOGS.keys()),
        help="Log key (varsayılan: pbxproj_conflict)",
    )

    # db-status
    subparsers.add_parser("db-status", help="Qdrant collection kayıt sayılarını göster")

    # pending
    subparsers.add_parser("pending", help="Pending analizleri listele")

    # promote
    promote_parser = subparsers.add_parser("promote", help="Pending kaydı historical_fixes'a taşı")
    promote_parser.add_argument("run_id", type=int, help="Promote edilecek run_id")
    promote_parser.add_argument("--pr-title", default="", help="PR başlığı (opsiyonel)")

    args = parser.parse_args()

    if args.command == "analyze":
        run_analyze(
            run_id=args.run_id,
            platform=args.platform,
            app=args.app,
            environment=args.environment,
            reason=args.reason,
            log_file=args.log_file,
        )
    elif args.command == "demo" or args.command is None:
        log_key = getattr(args, "log_key", "pbxproj_conflict")
        run_demo(log_key)
    elif args.command == "db-status":
        from qdrant_client import QdrantClient
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        client = QdrantClient(url=qdrant_url)
        collections = [
            "historical_fixes",
            "dependency_knowledge",
            "platform_knowledge",
            "conflict_resolution",
        ]
        print("\n📊 Qdrant Durumu")
        print(f"  URL : {qdrant_url}")
        print()
        print(f"  {'Collection':<28} {'Kayıt':>6}")
        print("  " + "-" * 36)
        total = 0
        for c in collections:
            try:
                n = client.count(c).count
            except Exception:
                n = "—"
            total += n if isinstance(n, int) else 0
            print(f"  {c:<28} {str(n):>6}")
        print("  " + "-" * 36)
        print(f"  {'TOPLAM':<28} {total:>6}\n")
    elif args.command == "pending":
        from .rag.pending_store import PendingStore

        store = PendingStore()
        items = store.list_pending()
        if not items:
            print("📭 Pending analiz yok.")
        else:
            print(f"\n📋 Pending Analizler ({len(items)} kayıt):")
            print("-" * 50)
            for item in items:
                print(
                    f"  run_id={item['run_id']}  "
                    f"branch={item.get('branch', '?')}  "
                    f"category={item.get('main_category', '?')}  "
                    f"confidence={item.get('confidence', 0):.2f}  "
                    f"created={item.get('created_at', '?')[:19]}"
                )
    elif args.command == "promote":
        from .feedback_service import promote_by_run_id

        result = promote_by_run_id(
            run_id=args.run_id,
            pr_title=args.pr_title,
        )
        if result["promoted"]:
            print(f"✅ run_id={args.run_id} → historical_fixes'a promote edildi.")
        else:
            print(f"❌ {result['message']}")


if __name__ == "__main__":
    main()