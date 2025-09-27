import SwiftUI

@main
struct AidayTradingApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var session = SessionStore()
    @StateObject private var notificationController = NotificationController()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(session)
                .environmentObject(notificationController)
                .onChange(of: scenePhase) { _, newPhase in
                    session.handleScenePhaseChange(newPhase)
                }
                .onAppear {
                    appDelegate.notificationController = notificationController
                    notificationController.updateSessionState(session.state)
                }
                .onChange(of: session.state) { _, newState in
                    notificationController.updateSessionState(newState)
                }
        }
    }
}
