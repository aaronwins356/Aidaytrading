import SwiftUI

struct AdminView: View {
    @EnvironmentObject private var notificationManager: NotificationManager
    @StateObject private var viewModel: AdminViewModel
    @State private var selectedUser: AdminUser?
    @State private var activeAlert: AdminViewModel.Alert?
    @State private var pendingMode: BotMode?
    @State private var showModeConfirmation = false

    init(
        context: UserSessionContext,
        repository: AdminRepository = AdminRepositoryImpl(),
        changeLogRepository: AdminChangeLogRepositoryProtocol = AdminChangeLogRepository()
    ) {
        _viewModel = StateObject(
            wrappedValue: AdminViewModel(
                actor: context.profile.username,
                repository: repository,
                changeLogRepository: changeLogRepository
            )
        )
    }

    init(viewModel: AdminViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        NavigationStack {
            List {
                botSection
                riskSection
                userSection
                changeLogSection
            }
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
            .background(Theme.background.ignoresSafeArea())
            .navigationTitle("Admin")
            .onAppear {
                viewModel.attach(notificationManager: notificationManager)
                viewModel.onAppear()
            }
            .alert(item: $activeAlert) { alert in
                Alert(title: Text(alert.title), message: Text(alert.message), dismissButton: .default(Text("OK")))
            }
            .confirmationDialog(
                "Switching to live mode will use real funds. Are you sure?",
                isPresented: $showModeConfirmation,
                presenting: pendingMode
            ) { mode in
                Button("Switch to Live", role: .destructive) {
                    guard let mode else { return }
                    Task { await viewModel.botControlViewModel.setMode(mode) }
                    pendingMode = nil
                }
                Button("Cancel", role: .cancel) { pendingMode = nil }
            }
            .sheet(item: $selectedUser) { user in
                UserDetailView(viewModel: viewModel.userManagementViewModel, user: user) {
                    selectedUser = nil
                }
            }
        }
        .onChange(of: viewModel.alert) { _, alert in
            guard let alert else { return }
            activeAlert = alert
            viewModel.alert = nil
        }
        .onChange(of: viewModel.riskViewModel.errorMessage) { _, message in
            guard let message else { return }
            activeAlert = .init(title: "Risk Guardrails", message: message)
            viewModel.riskViewModel.errorMessage = nil
        }
        .onChange(of: viewModel.userManagementViewModel.errorMessage) { _, message in
            guard let message else { return }
            activeAlert = .init(title: "User Management", message: message)
            viewModel.userManagementViewModel.errorMessage = nil
        }
        .onChange(of: viewModel.botControlViewModel.errorMessage) { _, message in
            guard let message else { return }
            activeAlert = .init(title: "Bot Control", message: message)
            viewModel.botControlViewModel.errorMessage = nil
        }
    }

    private var botSection: some View {
        Section {
            BotControlCard(
                status: viewModel.botControlViewModel.status,
                isBusy: viewModel.botControlViewModel.isBusy,
                onStart: { Task { await viewModel.botControlViewModel.startBot() } },
                onStop: { Task { await viewModel.botControlViewModel.stopBot() } },
                onModeChange: { mode in
                    if mode == .live {
                        pendingMode = mode
                        showModeConfirmation = true
                    } else {
                        Task { await viewModel.botControlViewModel.setMode(mode) }
                    }
                }
            )
            if viewModel.botControlViewModel.isLoading {
                ProgressView("Loading bot status…")
                    .frame(maxWidth: .infinity, alignment: .center)
            }
            if let message = viewModel.botControlViewModel.bannerMessage {
                Label(message, systemImage: "checkmark.circle")
                    .foregroundColor(Theme.accentGreen)
                    .onAppear {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                            viewModel.botControlViewModel.bannerMessage = nil
                        }
                    }
            }
        }
    }

    private var riskSection: some View {
        Section(header: Text("Risk Guardrails")) {
            if viewModel.riskViewModel.isLoading {
                ProgressView("Loading guardrails…")
            } else {
                RiskParameterCard(
                    title: "Max Drawdown",
                    subtitle: "Hard stop before capital breach",
                    value: viewModel.riskViewModel.binding(
                        for: \.maxDrawdownPercent,
                        range: RiskConfiguration.ranges.maxDrawdownPercent
                    ),
                    range: RiskConfiguration.ranges.maxDrawdownPercent,
                    step: 1
                ) { value in
                    "\(Int(value))%"
                }
                RiskParameterCard(
                    title: "Daily Loss Limit",
                    subtitle: "Circuit breaker for intraday loss",
                    value: viewModel.riskViewModel.binding(
                        for: \.dailyLossLimitPercent,
                        range: RiskConfiguration.ranges.dailyLossLimitPercent
                    ),
                    range: RiskConfiguration.ranges.dailyLossLimitPercent,
                    step: 0.5
                ) { value in
                    "\(String(format: "%.1f", value))%"
                }
                RiskParameterCard(
                    title: "Risk Per Trade",
                    subtitle: "Capital at risk per position",
                    value: viewModel.riskViewModel.binding(
                        for: \.riskPerTrade,
                        range: RiskConfiguration.ranges.riskPerTrade
                    ),
                    range: RiskConfiguration.ranges.riskPerTrade,
                    step: 0.001
                ) { value in
                    "\(String(format: "%.2f", value * 100))%"
                }
                RiskIntegerCard(
                    title: "Max Open Positions",
                    subtitle: "Concurrent strategies allowed",
                    value: viewModel.riskViewModel.positionsBinding(),
                    range: Int(RiskConfiguration.ranges.maxOpenPositions.lowerBound)...Int(RiskConfiguration.ranges.maxOpenPositions.upperBound)
                )
                RiskParameterCard(
                    title: "ATR Stop Multiplier",
                    subtitle: "Volatility-adjusted protection",
                    value: viewModel.riskViewModel.binding(
                        for: \.atrStopLossMultiplier,
                        range: RiskConfiguration.ranges.atrStopLossMultiplier
                    ),
                    range: RiskConfiguration.ranges.atrStopLossMultiplier,
                    step: 0.1
                ) { value in
                    "x\(String(format: "%.1f", value))"
                }
                RiskParameterCard(
                    title: "ATR Take-Profit Multiplier",
                    subtitle: "Scale exits with volatility",
                    value: viewModel.riskViewModel.binding(
                        for: \.atrTakeProfitMultiplier,
                        range: RiskConfiguration.ranges.atrTakeProfitMultiplier
                    ),
                    range: RiskConfiguration.ranges.atrTakeProfitMultiplier,
                    step: 0.1
                ) { value in
                    "x\(String(format: "%.1f", value))"
                }
                if let message = viewModel.riskViewModel.successMessage {
                    Label(message, systemImage: "checkmark.seal")
                        .foregroundColor(Theme.accentGreen)
                        .onAppear {
                            DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                                viewModel.riskViewModel.dismissSuccess()
                            }
                        }
                }
                Button {
                    Task { await viewModel.riskViewModel.save() }
                } label: {
                    Label("Save guardrails", systemImage: "square.and.arrow.down")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(AdminFilledButtonStyle(color: Theme.accentGreen))
                .disabled(!viewModel.riskViewModel.hasChanges || viewModel.riskViewModel.isSaving)
            }
        }
    }

    private var userSection: some View {
        Section(header: Text("User Management")) {
            if viewModel.userManagementViewModel.isLoading {
                ProgressView("Loading users…")
            } else {
                ForEach(viewModel.userManagementViewModel.users, id: \.id) { user in
                    UserRowView(user: user)
                        .contentShape(Rectangle())
                        .onTapGesture {
                            selectedUser = user
                        }
                        .swipeActions(edge: .trailing) {
                            if user.status == .pending {
                                Button("Approve") {
                                    Task { await viewModel.userManagementViewModel.approve(user) }
                                }
                                .tint(Theme.accentGreen)
                            }
                            if user.status == .disabled {
                                Button("Activate") {
                                    Task { await viewModel.userManagementViewModel.activate(user) }
                                }
                                .tint(.blue)
                            } else {
                                Button("Disable", role: .destructive) {
                                    Task { await viewModel.userManagementViewModel.disable(user) }
                                }
                            }
                        }
                }
                if let message = viewModel.userManagementViewModel.bannerMessage {
                    Label(message, systemImage: "checkmark.circle")
                        .foregroundColor(Theme.accentGreen)
                        .onAppear {
                            DispatchQueue.main.asyncAfter(deadline: .now() + 4) {
                                viewModel.userManagementViewModel.bannerMessage = nil
                            }
                        }
                }
            }
        }
    }

    private var changeLogSection: some View {
        Section(header: Text("Last Changes")) {
            ChangeLogList(entries: viewModel.changeLog)
                .listRowInsets(EdgeInsets())
                .listRowBackground(Color.clear)
            Button("Refresh history") {
                viewModel.refreshChangeLog()
            }
            .buttonStyle(.bordered)
        }
    }
}
