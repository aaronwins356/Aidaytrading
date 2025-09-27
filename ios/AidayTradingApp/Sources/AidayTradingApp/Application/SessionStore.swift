import Foundation
import SwiftUI

@MainActor
final class SessionStore: ObservableObject {
    enum SessionState: Equatable {
        case loading
        case loggedOut
        case pendingApproval(email: String)
        case authenticated(UserSessionContext)

        static func == (lhs: SessionStore.SessionState, rhs: SessionStore.SessionState) -> Bool {
            switch (lhs, rhs) {
            case (.loading, .loading), (.loggedOut, .loggedOut):
                return true
            case let (.pendingApproval(lEmail), .pendingApproval(rEmail)):
                return lEmail == rEmail
            case let (.authenticated(lContext), .authenticated(rContext)):
                return lContext.profile.id == rContext.profile.id && lContext.profile.role == rContext.profile.role
            default:
                return false
            }
        }
    }

    struct SessionError: Identifiable {
        let id = UUID()
        let message: String
    }

    @Published private(set) var state: SessionState
    @Published var error: SessionError?

    private let authService: AuthServiceProtocol
    private let tokenStorage: SecureTokenStorage
    private let biometricAuthenticator: BiometricAuthenticating
    private let idleManager: IdleTimeoutHandling
    private var hasBootstrapped = false
    private var isBiometricallyVerified = false

    init(
        authService: AuthServiceProtocol = AuthService(),
        tokenStorage: SecureTokenStorage = KeychainStorage(),
        biometricAuthenticator: BiometricAuthenticating = BiometricAuthenticator(),
        idleManager: IdleTimeoutHandling = IdleTimeoutManager(timeout: 15 * 60),
        previewState: SessionState = .loading
    ) {
        self.authService = authService
        self.tokenStorage = tokenStorage
        self.biometricAuthenticator = biometricAuthenticator
        self.idleManager = idleManager
        self.state = previewState
        self.idleManager.onTimeout = { [weak self] in
            Task { @MainActor in
                self?.handleTimeout()
            }
        }
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
                try await enforceBiometricAuthentication()
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
            state = .pendingApproval(email: profile.email)
        } catch {
            await handleError(error)
            state = .loggedOut
        }
    }

    func login(email: String, password: String) async {
        state = .loading
        do {
            let context = try await authService.login(email: email, password: password)
            try tokenStorage.save(tokens: context.tokens)
            try await enforceBiometricAuthentication()
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
                updateState(with: profile, tokens: tokens)
            } catch {
                await handleError(error)
            }
        }
    }

    func logout() {
        try? tokenStorage.delete()
        idleManager.stop()
        isBiometricallyVerified = false
        state = .loggedOut
    }

    func handleScenePhaseChange(_ phase: ScenePhase) {
        switch phase {
        case .active:
            if case .authenticated = state {
                idleManager.start()
            }
        case .background:
            idleManager.stop()
        default:
            break
        }
    }

    func registerInteraction() {
        idleManager.reset()
    }

    private func updateState(with profile: UserProfile, tokens: AuthTokens) {
        switch profile.approvalStatus {
        case .approved:
            let context = UserSessionContext(profile: profile, tokens: tokens)
            state = .authenticated(context)
            idleManager.start()
        case .pending:
            state = .pendingApproval(email: profile.email)
        case .rejected:
            state = .loggedOut
            error = SessionError(message: "Your account request was rejected. Contact support for assistance.")
        }
    }

    private func handleTimeout() {
        logout()
        error = SessionError(message: "You have been logged out due to inactivity.")
    }

    private func enforceBiometricAuthentication() async throws {
        guard !isBiometricallyVerified else { return }
        try await biometricAuthenticator.authenticate(reason: "Authenticate to access trading data")
        isBiometricallyVerified = true
    }

    private func handleError(_ error: Error) async {
        if let apiError = error as? APIErrorResponse {
            self.error = SessionError(message: apiError.message)
        } else if let localized = error as? LocalizedError, let message = localized.errorDescription {
            self.error = SessionError(message: message)
        } else {
            self.error = SessionError(message: "An unexpected error occurred. Please try again.")
        }
    }
}
