export default function replaceKeyInObjects<T extends Record<string, unknown>>(
  arr: T[] | undefined | null,
  oldKey: string,
  newKey: string
): T[] {
  if (!arr || !Array.isArray(arr) || arr.length === 0) {
    return [];
  }
  return arr.map(obj => {
    if (oldKey in obj) {
      const { [oldKey]: value, ...rest } = obj;
      return { ...rest, [newKey]: value } as T;
    }
    return obj;
  });
}
