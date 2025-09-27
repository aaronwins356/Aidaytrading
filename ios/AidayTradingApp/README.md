# AidayTrading iOS Client

This SwiftUI application provides the secure mobile entry point for the AidayTrading platform. It focuses on authentication, approval workflows, and role-aware navigation while enforcing strong client-side security controls.

## Features

- Signup and login flows with password strength validation (minimum 8 characters, upper/lowercase, and numeric characters).
- Account approval status detection with a dedicated pending screen after registration.
- JWT and refresh-token management stored securely in the system Keychain.
- Automatic session bootstrap with refresh token handling and biometric re-authentication.
- Role-based UI that surfaces Admin tooling only for approved administrators.
- Sensitive screen protections including Face ID / Touch ID gating, idle-timeout logout, and screen capture monitoring.
- Guided password reset experience wired to `/auth/forgot-password` and delivered via secure email.
- Unit and integration tests covering authentication, token handling, role detection, and approval workflow.
- Viewer dashboards with an equity curve chart, calendar heatmap computed in Central Time, and a paginated trade ledger.
- Data refresh every ten minutes using authenticated REST polling with offline cache fallbacks.
- Push notifications delivered via Firebase Cloud Messaging, including deep linking into the correct tab when tapped.
- Local notification fallback that warns the operator if the realtime feed stalls.

## Project structure

```
ios/AidayTradingApp
├── project.yml                 # XcodeGen definition for the Xcode project
├── README.md
├── Sources
│   └── AidayTradingApp
│       ├── Application         # App entry point & session store
│       ├── Features            # SwiftUI feature modules (Auth, Main, Pending approval)
│       ├── Services            # API client, endpoints, and models
│       └── Utilities           # Keychain, biometrics, password validation, idle timer
└── Tests
    └── AidayTradingAppTests    # XCTest targets with mocks and coverage for critical flows
```

## Viewer dashboards

- **Dashboard tab** – Presents the latest system status, balance, P/L metrics, and equity curve using the finance-grade theme in `AidaytradingApp/Foundation/DesignSystem`. Data loads instantly from disk cache and refreshes every ten minutes through `HomeViewModel`.
- **Calendar tab** – Uses `CalendarViewModel` and `CalendarPnLComputer` to group closed trades into Central Time day buckets. Tapping a day reveals trade-level details.
- **Trades tab** – Driven by `TradesViewModel` with client-side filters (symbol, outcome, date range) and infinite scroll pagination backed by `TradesRepositoryImpl`.
- Snapshot test fixtures in `Tests/AidayTradingAppTests/UI/Snapshot` render light-weight previews for the Dashboard and Calendar tabs.

## Calendar PnL computation

- `Domain/Repositories/CalendarPnLComputer.swift` aggregates closed trades by Central Time midnight using timezone-safe helpers.
- Open trades (`closed_at == nil`) are ignored until they settle.
- Unit tests in `CalendarPnLComputerTests` cover timezone boundaries and edge cases.

## Trades filters & pagination

- `TradesListView` hosts `FiltersView`, allowing symbol selection, win/loss toggles, and optional date range filtering.
- `TradesViewModel` keeps an in-memory trade cache, applies filters client-side, and requests the next page when the user nears the end of the list.

## Polling & caching

- `HomeViewModel` starts a ten-minute polling cycle defined by `AppConfig.pollingInterval` and cancels the task on disappear.
- Network fetches are persisted to disk through `DiskCache` (`equity_latest.json`, `profit_latest.json`, `trades_page_1.json`) and replayed immediately on launch.

## Accessibility & localization

- All metrics and heatmap cells expose VoiceOver labels describing amounts and sentiment.
- Dynamic Type and right-to-left layout are supported across the dashboard module.
- Currency formatting respects the user's locale while forcing USD as the currency code, and percent formatting includes explicit sign handling for clarity.

## Getting started

1. Install [XcodeGen](https://github.com/yonaskolb/XcodeGen) if you prefer generating the Xcode project from configuration:

   ```bash
   brew install xcodegen
   ```

2. Generate the Xcode project:

   ```bash
   cd ios/AidayTradingApp
   xcodegen generate
   ```

3. Open the project in Xcode and configure your signing team if required:

   ```bash
   open AidayTradingApp.xcodeproj
   ```

4. Create a `.xcconfig` or use Xcode build settings to inject the backend base URL if it differs from the default (`https://api.aidaytrading.com`).

## Backend integration

The app expects the backend (Prompt A) to expose HTTPS endpoints:

- `POST /auth/signup` → `{ "user": { ... } }`
- `POST /auth/login`  → `{ "tokens": { ... }, "user": { ... } }`
- `POST /auth/refresh`
- `GET /users/me`
- `GET /ws/status` (authenticated WebSocket channel publishing `{ "running": bool, "uptime_seconds": int }` updates)
- `GET /ws/equity` (authenticated WebSocket channel emitting equity curve points as `[timestamp, equity]` arrays at 10 minute cadence)
- `GET /ws/trades` (authenticated WebSocket channel streaming trade executions in real-time)
- `POST /api/v1/notifications/devices` expecting `{ "token": string, "platform": "ios", "timezone": string }`
- Remote notification payloads should include a `target` field (`home`, `trades`, `calendar`, or `admin`) so the app can deep link when the user taps alerts. Schedule balance/PnL recaps at 8am, 2pm, 8pm, and 2am Central Time and send bot state changes via FCM topics or user-specific tokens.

Responses should encode dates using ISO-8601 to match the bundled `JSONDecoder` configuration.

## Security considerations

- Tokens are persisted using the Keychain with the `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly` protection class.
- Face ID / Touch ID is required immediately after login or when restoring an existing session.
- An idle timer signs users out after 15 minutes of inactivity.
- Sensitive screens are blurred if a screen capture is detected, discouraging screenshots.
- The `APIClient` refuses to execute non-HTTPS requests.

## Testing & CI hooks

From the repository root you can execute unit tests via `xcodebuild` once the project is generated:

```bash
cd ios/AidayTradingApp
xcodegen generate
xcodebuild \
  -scheme AidayTradingApp \
  -destination 'platform=iOS Simulator,name=iPhone 15' \
  test
```

Static analysis can be plugged in using SwiftLint by adding a `.swiftlint.yml` configuration at the project root. Ensure your CI environment runs:

```bash
swiftlint
xcodebuild test
```

Automated tests use a mocked WebSocket server to validate reconnect logic, push-token registration, and notification routing behaviour. You can execute the tests via:

```bash
cd ios/AidayTradingApp
xcodegen generate
xcodebuild \
  -scheme AidayTradingApp \
  -destination 'platform=iOS Simulator,name=iPhone 15' \
  test
```

The mocked realtime layer lives in `Tests/AidayTradingAppTests/RealtimeServiceTests.swift`.

## Environment configuration

- Configure environment variables for any secrets (e.g., Brevo API key) at the backend level. The iOS client never stores or hardcodes sensitive values.
- Update `APIEnvironment.baseURL` to point to your deployment if it changes.
- Add your Firebase `GoogleService-Info.plist` under `Sources/AidayTradingApp/Application/` before building. The app registers for push notifications via Firebase Messaging and expects APNs credentials configured in the Firebase console.
- Enable the `Push Notifications` capability in your Apple Developer account and ensure the `remote-notification` background mode entitlement is granted.

## Next steps

- Wire dashboard, calendar, and trade list views to live backend data.
- Extend admin tooling with approval actions (approve/reject users, manage risk limits).
- Add push notification support for approval status changes.
- Expand dashboard analytics with additional strategy drill-downs.
- Add per-strategy real-time feeds and refine notification targeting for complex user roles.
