//! Pure navigation-decision logic for online inference.
//!
//! Deliberately free of any `esp-hal` / hardware dependency: it operates only on
//! drivable-area *fractions* already extracted from the segmentation mask, so the
//! inline `#[cfg(test)]` tests run on the host:
//!
//! ```text
//! rustc --test firmware/src/control.rs -o /tmp/control_test && /tmp/control_test
//! ```
//!
//! The positive class is **drivable / road** (matches `extract_binary_road` on the
//! Python side). The policy is intentionally conservative — the same "don't move
//! unless sure" stance behind the F0.5 selection metric: if the centre lane is not
//! clearly drivable we stop rather than guess.

#![allow(dead_code)]

/// Steering command derived from one frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Steer {
    Left,
    Center,
    Right,
}

/// A single navigation decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decision {
    /// `true` = drive, `false` = stop (centre not confidently drivable).
    pub go: bool,
    /// Suggested steering direction.
    pub steer: Steer,
}

/// Tunable thresholds for [`decide`]. Defaults are conservative.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    /// Minimum drivable fraction in the centre lane required to move at all.
    pub go_threshold: f32,
    /// How much more drivable one side must be than the other before we steer
    /// toward it (hysteresis to avoid twitching when sides are balanced).
    pub steer_margin: f32,
}

impl Default for Policy {
    fn default() -> Self {
        // Require the centre lane to be >40% drivable before moving, and only
        // steer when one side beats the other by >10 percentage points.
        Self { go_threshold: 0.40, steer_margin: 0.10 }
    }
}

/// Decide go/stop + steering from per-region drivable fractions.
///
/// `left`, `center`, `right` are the fraction (0.0..=1.0) of pixels classified
/// drivable in the left / centre / right thirds of the region of interest
/// (typically the lower band of the mask, where the near ground lies).
///
/// Policy:
/// * If the centre lane is below `go_threshold` → **stop** (`go = false`). We
///   still report the steering hint so a caller can creep/turn if it chooses.
/// * Otherwise **go**, steering toward whichever side is more open, but only if
///   it beats the other side by `steer_margin`; else hold centre.
pub fn decide(left: f32, center: f32, right: f32, policy: Policy) -> Decision {
    let go = center >= policy.go_threshold;

    let steer = if (left - right).abs() < policy.steer_margin {
        Steer::Center
    } else if left > right {
        Steer::Left
    } else {
        Steer::Right
    };

    Decision { go, steer }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_ahead_goes_straight() {
        let d = decide(0.8, 0.9, 0.8, Policy::default());
        assert!(d.go);
        assert_eq!(d.steer, Steer::Center);
    }

    #[test]
    fn blocked_center_stops() {
        let d = decide(0.7, 0.1, 0.7, Policy::default());
        assert!(!d.go);
    }

    #[test]
    fn steers_toward_open_side() {
        let left = decide(0.9, 0.6, 0.2, Policy::default());
        assert!(left.go);
        assert_eq!(left.steer, Steer::Left);

        let right = decide(0.2, 0.6, 0.9, Policy::default());
        assert_eq!(right.steer, Steer::Right);
    }

    #[test]
    fn balanced_sides_hold_center() {
        // Sides within steer_margin of each other -> Center even if both high.
        let d = decide(0.85, 0.9, 0.88, Policy::default());
        assert_eq!(d.steer, Steer::Center);
    }

    #[test]
    fn boundary_at_go_threshold_is_inclusive() {
        let p = Policy::default();
        let d = decide(0.5, p.go_threshold, 0.5, p);
        assert!(d.go, "center exactly at threshold should be drivable");
    }

    #[test]
    fn custom_policy_is_respected() {
        // Stricter threshold flips a borderline frame from go -> stop.
        let strict = Policy { go_threshold: 0.7, steer_margin: 0.1 };
        assert!(!decide(0.5, 0.5, 0.5, strict).go);
        assert!(decide(0.5, 0.5, 0.5, Policy::default()).go);
    }
}
