import SwiftUI

@main
struct AidayTradingApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var session = SessionStore()
    @StateObject private var notificationManager = NotificationManager()
    @StateObject private var realtimeClient = TradingWebSocketClient()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(session)
                .environmentObject(notificationManager)
                .environmentObject(realtimeClient)
                .onChange(of: scenePhase) { _, newPhase in
                    session.handleScenePhaseChange(newPhase)
                }
                .onAppear {
                    appDelegate.notificationManager = notificationManager
                    notificationManager.updateSessionState(session.state)
                    notificationManager.bind(to: realtimeClient)
                    updateRealtimeConnection(for: session.state)
                }
                .onChange(of: session.state) { _, newState in
                    notificationManager.updateSessionState(newState)
                    updateRealtimeConnection(for: newState)
                }
        }
    }

    private func updateRealtimeConnection(for state: SessionStore.SessionState) {
        switch state {
        case .authenticated(let context):
            realtimeClient.connect(accessToken: context.tokens.accessToken)
        default:
            realtimeClient.disconnect()
        }
    }
}
