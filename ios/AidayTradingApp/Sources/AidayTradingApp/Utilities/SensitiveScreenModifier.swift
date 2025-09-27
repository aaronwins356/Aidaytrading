import SwiftUI
import UIKit

struct SensitiveScreenModifier: ViewModifier {
    @State private var isCaptured = UIScreen.main.isCaptured

    func body(content: Content) -> some View {
        content
            .blur(radius: isCaptured ? 40 : 0)
            .overlay {
                if isCaptured {
                    VStack(spacing: 16) {
                        Image(systemName: "eye.slash")
                            .font(.largeTitle)
                        Text("Screen capture disabled")
                            .font(.headline)
                    }
                    .padding()
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
                }
            }
            .onReceive(NotificationCenter.default.publisher(for: UIScreen.capturedDidChangeNotification)) { _ in
                isCaptured = UIScreen.main.isCaptured
            }
    }
}

extension View {
    func sensitive() -> some View {
        modifier(SensitiveScreenModifier())
    }
}
