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
  22: '<path d="M20 49L58 20L105 47 M20 49L76 17L105 47"/><path d="M55 15L74 12" stroke="#c96f42"/><circle cx="58" cy="20" r="4" fill="#e8b08f"/><circle cx="76" cy="17" r="4" fill="#d8824c"/><path d="M15 48H27V54H15Z M98 46H110V52H98Z"/>',
  23: '<path d="M16 47Q61 -8 115 35" stroke-dasharray="3 3"/><ellipse cx="28" cy="36" rx="7" ry="4"/><ellipse cx="61" cy="18" rx="11" ry="5" fill="#b3c0e4" transform="rotate(-12 61 18)"/><ellipse cx="99" cy="25" rx="15" ry="6" transform="rotate(23 99 25)"/>',
  24: '<path d="M22 42L33 18L47 42Z M78 35L94 14L103 40Z"/><g fill="#889ad3"><circle cx="22" cy="42" r="4"/><circle cx="78" cy="35" r="4"/></g><g fill="#e4a179"><circle cx="33" cy="18" r="4"/><circle cx="94" cy="14" r="4"/></g><path d="M48 27L72 22" stroke-dasharray="3 3"/>',
  25: '<path d="M17 16H58V48H17Z M17 32H58 M37 16V48 M69 14Q86 25 110 16L110 48Q87 55 69 44Z M89 21L89 50 M69 30Q88 41 110 32"/><circle cx="89" cy="35" r="4" fill="#d99865"/>',
  26: '<path d="M14 48L52 20L95 47 M14 48L79 26L95 47" stroke-dasharray="3 3"/><path d="M32 38Q64 2 95 38" stroke="#526ec2"/><circle cx="52" cy="20" r="4" fill="#b7c2e1"/><circle cx="79" cy="26" r="4" fill="#da996d"/>',
  27: '<path d="M12 20H116V44H12Z"/><path d="M36 20V44 M60 20V44 M92 20V44 M108 20V44"/><path d="M13 43V21H35V43Z" fill="#b2bfda"/><path d="M61 43V21H91V43Z" fill="#d9aa84"/>',
  28: '<g fill="#97a8ce"><ellipse cx="27" cy="23" rx="11" ry="7"/><ellipse cx="41" cy="31" rx="14" ry="9"/><ellipse cx="23" cy="41" rx="8" ry="6"/></g><path d="M54 32H72 M67 27L73 32L67 37"/><g fill="#a2b18c"><ellipse cx="90" cy="24" rx="11" ry="7"/><ellipse cx="100" cy="34" rx="14" ry="9"/></g>',
  29: '<path d="M13 45Q56 2 117 35" stroke="#a8b3bf"/><path d="M17 41L37 26L58 21L80 22L102 29" stroke="#576fbb"/><g fill="#db905e"><rect x="13" y="37" width="8" height="8"/><rect x="76" y="18" width="8" height="8"/></g><circle cx="37" cy="26" r="3"/><circle cx="58" cy="21" r="3"/><circle cx="102" cy="29" r="3"/>',
  30: '<path d="M17 10V53H117 M79 13V52" stroke-dasharray="3 2"/><path d="M27 43L52 28L72 22L102 16"/><g fill="#b0bdd9"><circle cx="27" cy="43" r="4"/><circle cx="52" cy="28" r="4"/><circle cx="102" cy="16" r="4"/></g><circle cx="72" cy="22" r="6" fill="#e59961"/>',
};

export function sceneIllustration(number) {
  const drawing = silhouettes[number];
  return drawing && `<svg class="row-art" viewBox="0 0 130 65" aria-hidden="true"><g fill="none" stroke="#788ba1" stroke-width="1.7" stroke-linejoin="round">${drawing}</g></svg>`;
}
