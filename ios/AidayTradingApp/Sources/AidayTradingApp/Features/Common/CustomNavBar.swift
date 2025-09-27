import SwiftUI

struct CustomNavBar<Content: View>: View {
    let title: String
    @ViewBuilder var trailing: () -> Content

    init(title: String, @ViewBuilder trailing: @escaping () -> Content = { EmptyView() }) {
        self.title = title
        self.trailing = trailing
    }

    var body: some View {
        HStack {
            Text(title)
                .font(.system(size: 28, weight: .bold))
                .foregroundStyle(.white)
            Spacer()
            trailing()
        }
        .padding(.horizontal, 24)
        .padding(.top, 48)
        .padding(.bottom, 16)
        .background(Theme.background.opacity(0.95))
    }
}

extension View {
    func customNavigationBar<Trailing: View>(title: String, @ViewBuilder trailing: @escaping () -> Trailing = { EmptyView() }) -> some View {
        modifier(CustomNavBarModifier(title: title, trailing: trailing))
    }
}

private struct CustomNavBarModifier<Trailing: View>: ViewModifier {
    let title: String
    @ViewBuilder var trailing: () -> Trailing

    func body(content: Content) -> some View {
        VStack(spacing: 0) {
            CustomNavBar(title: title, trailing: trailing)
            content
        }
        .background(Theme.background)
    }
}
