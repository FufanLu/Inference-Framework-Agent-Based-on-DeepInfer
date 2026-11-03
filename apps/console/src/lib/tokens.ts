/** Make whitespace visible inside a token chip. */
export function visibleText(text: string): string {
  if (text === "") return "∅";
  return text.replace(/ /g, "␠").replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/\r/g, "␍");
}
