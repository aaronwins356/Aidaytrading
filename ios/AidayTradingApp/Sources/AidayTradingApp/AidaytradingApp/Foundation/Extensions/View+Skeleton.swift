import SwiftUI

struct SkeletonViewModifier: ViewModifier {
    @State private var phase: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .redacted(reason: .placeholder)
            .overlay(
                LinearGradient(
                    gradient: Gradient(colors: [Color.white.opacity(0.1), Color.white.opacity(0.4), Color.white.opacity(0.1)]),
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .rotationEffect(.degrees(20))
                .offset(x: phase * 200)
                .blendMode(.plusLighter)
            )
            .mask(content)
            .onAppear {
                withAnimation(.linear(duration: 1.2).repeatForever(autoreverses: false)) {
                    phase = 1
                }
            }
    }
}

extension View {
    func skeleton() -> some View {
        modifier(SkeletonViewModifier())
    }
}
