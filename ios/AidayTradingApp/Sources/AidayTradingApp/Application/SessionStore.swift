import Foundation
import SwiftUI

@MainActor
final class SessionStore: ObservableObject {
    enum SessionState: Equatable {
        case loading
        case loggedOut
        case pendingApproval(PendingApprovalContext)
        case authenticated(UserSessionContext)

        static func == (lhs: SessionStore.SessionState, rhs: SessionStore.SessionState) -> Bool {
            switch (lhs, rhs) {
            case (.loading, .loading), (.loggedOut, .loggedOut):
                return true
            case let (.pendingApproval(left), .pendingApproval(right)):
                return left == right
            case let (.authenticated(lContext), .authenticated(rContext)):
                return lContext.profile.id == rContext.profile.id && lContext.profile.role == rContext.profile.role
            default:
                return false
            }
        }
    }

    struct PendingApprovalContext: Equatable {
        let username: String
        let email: String
        var tokens: AuthTokens?
    }

    struct SessionAlert: Identifiable {
        let id = UUID()
        let title: String
        let message: String
    }

    @Published private(set) var state: SessionState
    @Published var alert: SessionAlert?

    private let authService: AuthServiceProtocol
    private let tokenStorage: SecureTokenStorage
    private let biometricAuthenticator: BiometricAuthenticating
    private let idleManager: IdleTimeoutHandling
    private let approvalService: ApprovalServiceProtocol
    private var hasBootstrapped = false
    private var isBiometricallyVerified = false
    private var approvalPollingTask: Task<Void, Never>?
    private var pendingApprovalContext: PendingApprovalContext?

    init(
        authService: AuthServiceProtocol = AuthService(),
        tokenStorage: SecureTokenStorage = KeychainStorage(),
        biometricAuthenticator: BiometricAuthenticating = BiometricAuthenticator(),
        idleManager: IdleTimeoutHandling = IdleTimeoutManager(timeout: 15 * 60),
        approvalService: ApprovalServiceProtocol = ApprovalService(),
        previewState: SessionState = .loading
    ) {
        self.authService = authService
        self.tokenStorage = tokenStorage
        self.biometricAuthenticator = biometricAuthenticator
        self.idleManager = idleManager
        self.approvalService = approvalService
        self.state = previewState
        self.idleManager.onTimeout = { [weak self] in
            Task { @MainActor in
                self?.handleTimeout()
            }
        }
    }

    deinit {
        approvalPollingTask?.cancel()
    }

    func bootstrap() async {
        guard !hasBootstrapped else { return }
        hasBootstrapped = true
        state = .loading
        do {
            if var tokens = try tokenStorage.load() {
                if !tokens.isAccessTokenValid {
                    tokens = try await authService.refresh(using: tokens.refreshToken)
                    try tokenStorage.save(tokens: tokens)
                }
                let profile = try await authService.loadProfile(accessToken: tokens.accessToken)
                if profile.approvalStatus == .approved {
                    try await enforceBiometricAuthentication()
                }
                updateState(with: profile, tokens: tokens)
            } else {
                state = .loggedOut
            }
        } catch {
            await handleError(error)
            state = .loggedOut
        }
    }

    func signup(username: String, email: String, password: String) async {
        state = .loading
        do {
            let profile = try await authService.signup(username: username, email: email, password: password)
            let context = PendingApprovalContext(username: profile.username, email: profile.email, tokens: nil)
            startApprovalPolling(with: context)
        } catch {
            await handleError(error)
            state = .loggedOut
        }
    }

    func login(username: String, password: String) async {
        state = .loading
        do {
            let context = try await authService.login(username: username, password: password)
            try tokenStorage.save(tokens: context.tokens)
            if context.profile.approvalStatus == .approved {
                try await enforceBiometricAuthentication()
            }
            updateState(with: context.profile, tokens: context.tokens)
        } catch {
            try? tokenStorage.delete()
            await handleError(error)
            state = .loggedOut
        }
    }

    func refreshProfile() {
        Task {
            do {
                guard var tokens = try tokenStorage.load() else {
                    return
                }
                if !tokens.isAccessTokenValid {
                    tokens = try await authService.refresh(using: tokens.refreshToken)
                    try tokenStorage.save(tokens: tokens)
                }
                let profile = try await authService.loadProfile(accessToken: tokens.accessToken)
                if profile.approvalStatus == .approved {
                    try await enforceBiometricAuthentication()
                }
                updateState(with: profile, tokens: tokens)
            } catch {
                await handleError(error)
            }
        }
    }

    func requestPasswordReset(email: String) {
        Task {
            do {
                try await authService.requestPasswordReset(email: email)
                await MainActor.run {
                    self.alert = SessionAlert(title: "Reset email sent", message: "If an account exists for \(email), instructions are on the way.")
                }
            } catch {
                await handleError(error)
            }
        }
    }

    func logout() {
        Task {
            if let tokens = try? tokenStorage.load() {
                try? await authService.logout(accessToken: tokens.accessToken)
            }
            try? tokenStorage.delete()
            await MainActor.run {
                self.stopApprovalPolling()
                self.idleManager.stop()
                self.isBiometricallyVerified = false
                self.state = .loggedOut
            }
        }
    }

    func handleScenePhaseChange(_ phase: ScenePhase) {
        switch phase {
        case .active:
            if case .authenticated = state {
                idleManager.start()
                Task { try? await enforceBiometricAuthentication() }
            }
        case .background, .inactive:
            isBiometricallyVerified = false
            idleManager.stop()
        default:
            break
        }
    }

    func registerInteraction() {
        if case .authenticated = state {
            idleManager.reset()
        }
    }

    private func updateState(with profile: UserProfile, tokens: AuthTokens) {
        switch profile.approvalStatus {
        case .approved:
            stopApprovalPolling()
            let context = UserSessionContext(profile: profile, tokens: tokens)
            state = .authenticated(context)
            idleManager.start()
        case .pending:
            let context = PendingApprovalContext(username: profile.username, email: profile.email, tokens: tokens)
            startApprovalPolling(with: context)
        case .rejected:
            stopApprovalPolling()
            state = .loggedOut
            alert = SessionAlert(title: "Access denied", message: "Your account request was rejected. Contact support for assistance.")
        }
    }

    private func handleTimeout() {
        logout()
        alert = SessionAlert(title: "Session expired", message: "You have been logged out due to inactivity.")
    }

    private func enforceBiometricAuthentication() async throws {
        guard !isBiometricallyVerified else { return }
        try await biometricAuthenticator.authenticate(reason: "Authenticate to access trading data")
        isBiometricallyVerified = true
    }

    private func startApprovalPolling(with context: PendingApprovalContext) {
        pendingApprovalContext = context
        state = .pendingApproval(context)
        approvalPollingTask?.cancel()
        approvalPollingTask = Task { [weak self] in
            await self?.pollApprovalStatus()
        }
    }

    private func stopApprovalPolling() {
        approvalPollingTask?.cancel()
        approvalPollingTask = nil
        pendingApprovalContext = nil
    }

    private func pollApprovalStatus() async {
        while !Task.isCancelled {
            guard let context = await MainActor.run(body: { self.pendingApprovalContext }) else {
                return
            }
            await checkApprovalStatus(for: context)
            try? await Task.sleep(nanoseconds: 30 * NSEC_PER_SEC)
        }
    }

    private func checkApprovalStatus(for context: PendingApprovalContext) async {
        do {
            if var tokens = context.tokens {
                if !tokens.isAccessTokenValid {
                    tokens = try await authService.refresh(using: tokens.refreshToken)
                    try tokenStorage.save(tokens: tokens)
                    await MainActor.run {
                        self.pendingApprovalContext?.tokens = tokens
                    }
                }
                let profile = try await authService.loadProfile(accessToken: tokens.accessToken)
                if profile.approvalStatus == .approved {
                    await MainActor.run {
                        self.updateState(with: profile, tokens: tokens)
                    }
                } else if profile.approvalStatus == .rejected {
                    await MainActor.run {
                        self.stopApprovalPolling()
                        self.state = .loggedOut
                        self.alert = SessionAlert(title: "Access denied", message: "Your account request was rejected. Contact support for assistance.")
                    }
                }
            } else {
                let status = try await approvalService.fetchStatus(username: context.username, email: context.email)
                await MainActor.run {
                    switch status {
                    case .approved:
                        self.stopApprovalPolling()
                        self.state = .loggedOut
                        self.alert = SessionAlert(title: "Account approved", message: "Your account has been approved. Please sign in to continue.")
                    case .rejected:
                        self.stopApprovalPolling()
                        self.state = .loggedOut
                        self.alert = SessionAlert(title: "Access denied", message: "Your account request was rejected. Contact support for assistance.")
                    case .pending:
                        break
                    }
                }
            }
        } catch {
            await handleError(error)
        }
    }

    private func handleError(_ error: Error) async {
        if let apiError = error as? APIErrorResponse {
            self.alert = SessionAlert(title: "Request failed", message: apiError.message)
        } else if let localized = error as? LocalizedError, let message = localized.errorDescription {
            self.alert = SessionAlert(title: "Request failed", message: message)
        } else {
            self.alert = SessionAlert(title: "Unexpected error", message: "An unexpected error occurred. Please try again.")
        }
    }
}
