import Foundation

@MainActor
protocol AdminActionRecording: AnyObject {
    func recordChange(
        category: AdminChangeLogEntry.Category,
        summary: String,
        details: String,
        payload: [String: Any],
        showBanner: Bool
    )
}

@MainActor
final class AdminViewModel: ObservableObject, AdminActionRecording {
    struct Alert: Identifiable {
        let id = UUID()
        let title: String
        let message: String
    }

    @Published private(set) var changeLog: [AdminChangeLogEntry] = []
    @Published var alert: Alert?

    let riskViewModel: RiskViewModel
    let userManagementViewModel: UserManagementViewModel
    let botControlViewModel: BotControlViewModel

    private let actor: String
    private let changeLogRepository: AdminChangeLogRepositoryProtocol
    private var notificationManager: NotificationManager?

    init(
        actor: String,
        repository: AdminRepository = AdminRepositoryImpl(),
        changeLogRepository: AdminChangeLogRepositoryProtocol = AdminChangeLogRepository()
    ) {
        self.actor = actor
        self.changeLogRepository = changeLogRepository
        self.riskViewModel = RiskViewModel(repository: repository)
        self.userManagementViewModel = UserManagementViewModel(repository: repository)
        self.botControlViewModel = BotControlViewModel(repository: repository)
        self.riskViewModel.actionRecorder = self
        self.userManagementViewModel.actionRecorder = self
        self.botControlViewModel.actionRecorder = self
    }

    func attach(notificationManager: NotificationManager) {
        guard self.notificationManager !== notificationManager else { return }
        self.notificationManager = notificationManager
    }

    func onAppear() {
        Task {
            await loadChangeLog()
        }
        Task { await riskViewModel.load() }
        Task { await userManagementViewModel.loadUsers() }
        Task { await botControlViewModel.loadStatus() }
    }

    func refreshChangeLog() {
        Task { await loadChangeLog() }
    }

    private func loadChangeLog() async {
        do {
            changeLog = try changeLogRepository.fetchLatest(limit: 10)
        } catch {
            alert = Alert(title: "Change log", message: "Failed to load change history: \(error.localizedDescription)")
        }
    }

    func recordChange(
        category: AdminChangeLogEntry.Category,
        summary: String,
        details: String,
        payload: [String: Any],
        showBanner: Bool
    ) {
        let entry = AdminChangeLogEntry(
            id: UUID(),
            timestamp: Date(),
            actor: actor,
            summary: summary,
            details: details,
            category: category
        )
        do {
            try changeLogRepository.record(entry)
            changeLog.insert(entry, at: 0)
            if changeLog.count > 10 {
                changeLog = Array(changeLog.prefix(10))
            }
        } catch {
            alert = Alert(title: "Change log", message: "Failed to persist change: \(error.localizedDescription)")
        }
        notificationManager?.recordAdminEvent(title: summary, body: details, payload: payload, showBanner: showBanner)
    }
}
