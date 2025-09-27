import Foundation

protocol IdleTimeoutHandling {
    var onTimeout: (() -> Void)? { get set }
    func start()
    func reset()
    func stop()
}

final class IdleTimeoutManager: IdleTimeoutHandling {
    private let timeout: TimeInterval
    private var timer: Timer?
    var onTimeout: (() -> Void)?

    init(timeout: TimeInterval) {
        self.timeout = timeout
    }

    func start() {
        reset()
    }

    func reset() {
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: timeout, repeats: false) { [weak self] _ in
            self?.onTimeout?()
        }
    }

    func stop() {
        timer?.invalidate()
        timer = nil
    }
}
