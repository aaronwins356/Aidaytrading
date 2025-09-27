import Foundation

@MainActor
final class BotControlViewModel: ObservableObject {
    @Published private(set) var status: BotStatus?
    @Published private(set) var isLoading = false
    @Published private(set) var isBusy = false
    @Published var errorMessage: String?
    @Published var bannerMessage: String?

    weak var actionRecorder: AdminActionRecording?

    private let repository: AdminRepository

    init(repository: AdminRepository) {
        self.repository = repository
    }

    func loadStatus() async {
        isLoading = true
        defer { isLoading = false }
        do {
            status = try await repository.fetchBotStatus()
        } catch {
            errorMessage = "Failed to load bot status: \(error.localizedDescription)"
        }
    }

    func startBot() async {
        guard !isBusy else { return }
        await performAction(summary: "Bot started", action: repository.startBot)
    }

    func stopBot() async {
        guard !isBusy else { return }
        await performAction(summary: "Bot stopped", action: repository.stopBot)
    }

    func setMode(_ mode: BotMode) async {
        guard !isBusy else { return }
        guard status?.mode != mode else { return }
        await performAction(summary: mode == .live ? "Switched to live" : "Switched to paper", action: { try await self.repository.setBotMode(mode) })
    }

    private func performAction(summary: String, action: @escaping () async throws -> BotStatus) async {
        isBusy = true
        bannerMessage = nil
        defer { isBusy = false }
        do {
            let updated = try await action()
            status = updated
            bannerMessage = "\(summary) at \(DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .short))"
            actionRecorder?.recordChange(
                category: .bot,
                summary: summary,
                details: "Mode: \(updated.mode.description), Running: \(updated.running ? "Yes" : "No")",
                payload: [
                    "running": updated.running,
                    "mode": updated.mode.rawValue,
                    "timestamp": updated.lastUpdated.timeIntervalSince1970
                ],
                showBanner: true
            )
        } catch {
            errorMessage = "Bot action failed: \(error.localizedDescription)"
        }
    }
}
