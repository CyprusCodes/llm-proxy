export default function replaceKeyInObjects<T extends Record<string, unknown>>(
  arr: T[],
  oldKey: string,
  newKey: string
): T[] {
  return arr.map((obj) => {
    if (oldKey in obj) {
      const { [oldKey]: value, ...rest } = obj;
      return { ...rest, [newKey]: value } as T;
    }
    return obj;
  });
}
