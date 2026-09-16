import Foundation

// Localization for strings the app builds ITSELF, rather than hands to
// SwiftUI as a literal.
//
// SwiftUI's `Text("...")` takes a LocalizedStringKey when it is given a
// literal, so most of the interface localizes with no code at all. Two
// shapes do not, and both look identical to the working ones:
//
//   * `Text("a" + "b")` is a String EXPRESSION, so it resolves to the
//     verbatim overload and is never looked up. Those sites wrap the
//     whole argument in LocalizedStringKey(...), which takes a String
//     at runtime and so may be concatenated freely.
//   * a String held in a model (a permission row's title, an error a
//     helper reported) reaches Text as a variable, which is the same
//     verbatim overload. Those localize HERE, where the string is
//     built, or not at all.
//
// `L` is the second case. It is deliberately terse: these call sites
// carry multi-line concatenated English, and NSLocalizedString's full
// spelling on each one would push the text itself off the line.
//
// The key IS the English text -- there is no separate identifier
// scheme. That keeps the sources readable and means a missing
// translation falls back to a correct English string rather than to a
// symbol like `perm.camera.title`. Keep the English in
// Localizations/en.lproj/Localizable.strings in step with the source.
func L(_ key: String) -> String {
    NSLocalizedString(key, comment: "")
}
