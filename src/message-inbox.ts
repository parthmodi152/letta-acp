/**
 * A queue of SDK messages for one reader at a time, whose waits can time out
 * without losing the message that arrives afterwards.
 *
 * Abandoning a pending `stream().next()` leaves its resolver registered with
 * the SDK, so the next message goes to a reader nobody is listening to. The
 * inbox withdraws a timed-out waiter instead, and the message stays queued.
 */
export class MessageInbox<T> {
  private readonly queue: T[] = [];
  private readonly waiters: Array<(value: IteratorResult<T, undefined>) => void> = [];
  /** Ends the current waits with null, as if they had timed out. */
  private readonly wakers = new Map<(value: IteratorResult<T, undefined>) => void, () => void>();
  private closed = false;

  /** Ends every pending `next()` with null; queued messages stay queued. */
  wake(): void {
    for (const wake of [...this.wakers.values()]) wake();
  }

  push(value: T): void {
    if (this.closed) return;
    const waiter = this.waiters.shift();
    if (waiter) waiter({ value, done: false });
    else this.queue.push(value);
  }

  /** Removes and returns everything queued. */
  drain(): T[] {
    return this.queue.splice(0);
  }

  close(): void {
    this.closed = true;
    for (const waiter of this.waiters.splice(0)) waiter({ value: undefined, done: true });
  }

  /**
   * The next message, `done` once the inbox is closed, or null when
   * `timeoutMs` passes first.
   */
  next(timeoutMs?: number): Promise<IteratorResult<T, undefined> | null> {
    const queued = this.queue.shift();
    if (queued !== undefined) return Promise.resolve({ value: queued, done: false });
    if (this.closed) return Promise.resolve({ value: undefined, done: true });
    return new Promise((resolve) => {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const waiter = (value: IteratorResult<T, undefined>) => {
        if (timer) clearTimeout(timer);
        this.wakers.delete(waiter);
        resolve(value);
      };
      const withdraw = () => {
        if (timer) clearTimeout(timer);
        this.wakers.delete(waiter);
        const index = this.waiters.indexOf(waiter);
        if (index >= 0) this.waiters.splice(index, 1);
        resolve(null);
      };
      this.waiters.push(waiter);
      this.wakers.set(waiter, withdraw);
      if (timeoutMs !== undefined) timer = setTimeout(withdraw, timeoutMs);
    });
  }
}
