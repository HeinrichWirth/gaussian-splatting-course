// Small diagrams for the lesson index. Coordinates share a 130 × 65 viewBox.
const silhouettes = {
  10: '<path d="M52 45V10 M52 45L90 38 M52 45L29 57 M52 45L87 16"/><circle cx="87" cy="16" r="4" fill="#e58b54"/>',
  11: '<path d="M12 33L109 9 M12 33L109 55 M53 10V55 M91 15V50"/><circle cx="53" cy="23" r="3"/><circle cx="91" cy="14" r="3"/>',
  12: '<g transform="rotate(-22 65 33)"><ellipse cx="65" cy="33" rx="40" ry="19"/><ellipse cx="65" cy="33" rx="14" ry="19"/><ellipse cx="65" cy="33" rx="40" ry="6"/></g>',
  13: '<ellipse cx="32" cy="31" rx="15" ry="21" transform="rotate(-28 32 31)"/><path d="M41 13L91 24 M23 49L88 43 M78 11L117 19L100 58L61 51Z"/><ellipse cx="89" cy="34" rx="15" ry="7" transform="rotate(-20 89 34)"/>',
  14: '<ellipse cx="76" cy="30" rx="22" ry="14" fill="#6678d2" opacity=".7"/><ellipse cx="53" cy="35" rx="23" ry="13" fill="#e68d57" opacity=".8"/>',
  15: '<circle cx="65" cy="32" r="17" fill="#e5e8d9"/><path d="M12 45L49 37 M111 11L81 24"/><circle cx="12" cy="45" r="4" fill="#e58b54"/><circle cx="111" cy="11" r="4" fill="#5872c5"/>',
  16: '<path d="M20 50L54 15L99 50 M20 50L69 31L99 50"/><circle cx="54" cy="15" r="4" fill="#e58b54"/><rect x="12" y="49" width="16" height="8"/><rect x="91" y="49" width="16" height="8"/>',
  17: '<ellipse cx="27" cy="32" rx="16" ry="12"/><path d="M47 32H65 M59 27L65 32L59 37"/><ellipse cx="83" cy="24" rx="12" ry="8"/><ellipse cx="99" cy="40" rx="12" ry="8"/>',
  18: '<rect x="27" y="8" width="78" height="48"/><path d="M53 8V56 M79 8V56 M27 24H105 M27 40H105"/><ellipse cx="66" cy="30" rx="22" ry="9" fill="#aab7df" opacity=".8" transform="rotate(-24 66 30)"/>',
  19: '<path d="M23 48L62 17L104 48 M23 48L72 29L104 48"/><circle cx="62" cy="17" r="4"/><circle cx="72" cy="29" r="3"/><path d="M13 47L29 51L25 56L12 53Z M111 47L96 51L100 56L113 53Z"/>',
  20: '<rect x="12" y="14" width="38" height="32"/><path d="M15 42L27 26L36 35L46 30 M56 30H69"/><g fill="#879b78"><ellipse cx="89" cy="20" rx="9" ry="6"/><ellipse cx="98" cy="32" rx="13" ry="8"/><ellipse cx="83" cy="41" rx="13" ry="6"/></g>',
  21: '<path d="M15 50V12 M15 50H115 M21 18L35 27L51 33L66 37L90 41L110 42" stroke="#5474c3"/><path d="M21 16L35 25L51 29L66 30L90 28L110 25" stroke="#d78656"/>',
};

export function sceneIllustration(number) {
  const drawing = silhouettes[number];
  return drawing && `<svg class="row-art" viewBox="0 0 130 65" aria-hidden="true"><g fill="none" stroke="#788ba1" stroke-width="1.7" stroke-linejoin="round">${drawing}</g></svg>`;
}
