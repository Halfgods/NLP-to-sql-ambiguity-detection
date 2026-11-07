/** A single, quiet blue glow on near-black — no rainbow blobs. */
export default function GradientBackdrop() {
  return (
    <div className="backdrop" aria-hidden="true">
      <span className="glow" />
      <span className="vignette" />
    </div>
  )
}
