import SwiftUI
import XCTest
@testable import AidayTradingApp

#if canImport(SwiftUI)

@MainActor
final class AuthenticationSnapshotTests: XCTestCase {
    func testLoginViewRendersSnapshot() throws {
        let view = LoginView(onSignup: {})
            .environmentObject(SessionStore(previewState: .loggedOut))
        let renderer = ImageRenderer(content: view.frame(width: 390, height: 844))

        #if canImport(UIKit)
        XCTAssertNotNil(renderer.uiImage)
        #elseif canImport(AppKit)
        XCTAssertNotNil(renderer.nsImage)
        #else
        throw XCTSkip("Snapshot rendering not supported on this platform")
        #endif
    }

    func testPendingApprovalViewRendersSnapshot() throws {
        let context = SessionStore.PendingApprovalContext(username: "snapshot", email: "snapshot@example.com", tokens: nil)
        let view = PendingApprovalView(context: context, onRefresh: {})
            .environmentObject(SessionStore(previewState: .pendingApproval(context)))
        let renderer = ImageRenderer(content: view.frame(width: 390, height: 844))

        #if canImport(UIKit)
        XCTAssertNotNil(renderer.uiImage)
        #elseif canImport(AppKit)
        XCTAssertNotNil(renderer.nsImage)
        #else
        throw XCTSkip("Snapshot rendering not supported on this platform")
        #endif
    }
}

#endif
