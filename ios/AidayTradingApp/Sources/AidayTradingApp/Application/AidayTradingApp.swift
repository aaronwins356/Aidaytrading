import SwiftUI

@main
struct AidayTradingApp: App {
    @StateObject private var session = SessionStore()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(session)
                .onChange(of: scenePhase) { _, newPhase in
                    session.handleScenePhaseChange(newPhase)
                }
        }
    }
}
