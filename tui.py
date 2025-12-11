import curses
import threading
import time
import textwrap
import locale

# Set locale for UTF-8 character handling
locale.setlocale(locale.LC_ALL, '')

# --- (Mock example) ---
class MockRagAgent:
    """Mock agent for TUI testing when no agent is provided."""
    def __init__(self):
        self.token_stats = {
            "total_tokens": 0,
            "time_elapsed": 0.0,
            "tokens_per_second": 0.0
        }

    def stream(self, question):
        """Simulates blocking RAG agent streaming."""
        start_time = time.time()
        token_count = 0
        response = (
            f"Hello! You asked: '{question}'. "
            "The status bar has been successfully moved to the bottom right of the chat window, "
            "just above the input separator line, providing fixed statistics visibility."
        )
        for word in response.split():
            yield word + " "
            token_count += 1
            time.sleep(0.04)
        yield "\n(Response complete.)"

        end_time = time.time()
        time_elapsed = end_time - start_time
        
        # Update internal stats after stream completion
        self.token_stats["total_tokens"] = token_count
        self.token_stats["time_elapsed"] = time_elapsed
        self.token_stats["tokens_per_second"] = token_count / time_elapsed if time_elapsed > 0 else 0.0

    def getStats(self):
        """Returns the calculated token stats."""
        return self.token_stats

class CursesTUI:
    """
    Terminal User Interface (TUI) class for the RAG Agent, 
    using curses for an interactive chat experience.
    """
    
    def __init__(self, ragAgent=None, ragName="Ragnarok"):
        """
        Initializes the TUI with the RAG agent instance. 
        Uses MockAgent if ragAgent is None.
        """
        self.ragAgent = ragAgent if ragAgent is not None else MockRagAgent()
        self.ragName = ragName
        
        # TUI State Variables
        self.chat_history = []
        self.input_buffer = ""
        self.streaming_active = False
        self.scroll_offset = 0 # Tracks how many lines UP the user has scrolled from the bottom
        self.last_chat_height = 0
        self.input_height = 3
        self.header_lines = 2 # Title (0) + Separator Line (1)
        self.performance_stats = 0.0 # Tokens per seconds
        self.status = "idle"

    def _init_colors(self):
        """Defines color pairs based on the dark style and assigns colors to prefixes."""
        curses.start_color()
        # curses.use_default_colors()
        
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK) # AI
        curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)  # YOU
        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK) 
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK) 

    def _get_color_pair_and_attr(self, sender, is_prefix=False):
        """Returns the curses color pair and attributes."""
        
        if is_prefix:
            if sender == 'You':
                return 2, curses.A_BOLD
            elif sender == self.ragName:
                return 3, curses.A_BOLD
            else:
                return 1, curses.A_NORMAL
        else:
            return 4, curses.A_NORMAL

    def _draw_header(self, stdscr, cols):
        """Draws the header (Title) and the separation line below it."""
        header_text = f"{self.ragName} Agent"
        
        # Row 0: Title
        try:
            stdscr.addstr(0, 1, header_text, curses.A_BOLD | curses.color_pair(1))
        except curses.error:
            pass
            
        # Row 1: Separation Line
        try:
            # stdscr.hline(1, 0, curses.ACS_HLINE, cols)
            stdscr.hline(1, 0, '-', cols)
        except curses.error:
            pass
            
        stdscr.refresh()

    def _draw_chat_window(self, chat_win, max_rows, cols):
        """Draws and updates the chat window, reserving the last row for status."""
        
        chat_win.clear()
        
        display_lines = []
        max_wrap_width = int(cols * 0.75) - 2 
        
        last_sender = None
        
        # 1. Process CHAT_HISTORY into display_lines (for wrapping)
        for sender, msg in self.chat_history:
            if last_sender == 'You' and sender == self.ragName:
                display_lines.append(('', '')) 
            
            prefix_text = f"[{sender}]: "
            content_width = max_wrap_width - len(prefix_text) if sender != 'You' else max_wrap_width
            wrapped_msg = textwrap.wrap(msg, content_width)
            
            prefix_color, prefix_attr = self._get_color_pair_and_attr(sender, is_prefix=True)
            content_color, content_attr = self._get_color_pair_and_attr(sender, is_prefix=False)
            
            for i, line in enumerate(wrapped_msg):
                line_parts = []
                if i == 0:
                    line_parts.append((prefix_text, prefix_color, prefix_attr))
                    line_parts.append((line, content_color, content_attr))
                else:
                    if sender != 'You':
                        line_parts.append((" " * len(prefix_text), content_color, content_attr))
                    line_parts.append((line, content_color, content_attr))
                
                display_lines.append((sender, line_parts))
                
            last_sender = sender 

        self.last_chat_height = len(display_lines) # Store total wrapped height
        
        # 2. Calculate Scroll Position
        # Reserve the last line for status, so chat content fits in max_rows - 1
        chat_content_rows = max_rows - 1 
        start_index_max = max(0, self.last_chat_height - chat_content_rows)
        
        # Clamp scroll_offset between 0 (bottom) and start_index_max (top)
        self.scroll_offset = max(0, min(start_index_max, self.scroll_offset))
        start_index = self.scroll_offset 
        
        # 3. Draw Lines
        for i in range(start_index, len(display_lines)):
            sender, line_parts = display_lines[i]
            y = i - start_index
            
            if y >= chat_content_rows: # Stop before the last reserved line
                break
                
            current_x = 1
            
            if sender == '' and line_parts == '':
                try: chat_win.addstr(y, current_x, "")
                except curses.error: pass
                continue

            total_line_width = sum(len(text) for text, _, _ in line_parts)
            
            if sender == 'You':
                current_x = cols - total_line_width - 1 
            else:
                current_x = 1
            
            for text, color_pair, attr in line_parts:
                try:
                    chat_win.addstr(y, current_x, text, attr | curses.color_pair(color_pair))
                    current_x += len(text)
                except curses.error:
                    break 
        
        # 4. Draw STATUS BAR on the last line of the chat window (y = max_rows - 1)
        
        # The maximum scrollable offset (the line index of the top-most visible line)
        max_scroll_offset = start_index_max 
        scroll_status = "BOTTOM" if self.scroll_offset == max_scroll_offset else "SCROLLED"
        
        stats_text = (
            f"[S:{self.status} t/s:{self.performance_stats:0.2f}] "
            f"[M:{len(self.chat_history)} H:{self.last_chat_height} S:{self.scroll_offset}/{max_scroll_offset}] "
            f"[{scroll_status}] "
        )
        
        status_y = max_rows - 1
        
        # Clear the status line completely before drawing (to remove residual text)
        try:
            chat_win.addstr(status_y, 0, " " * cols) 
        except curses.error:
            pass

        if len(stats_text) < cols - 2: 
            start_col = cols - len(stats_text) - 1
            # Draw stats with A_DIM attribute to simulate a less emphasized font
            try:
                chat_win.addstr(status_y, start_col, stats_text, curses.A_DIM | curses.color_pair(1)) 
            except curses.error:
                pass

        chat_win.refresh()
        
    def _draw_input_window(self, input_win, cols):
        """Draws the input window and a clean separator line."""
        input_win.clear()
        
        # Row 0 of input_win: Only a clean Separator Line
        # input_win.hline(0, 0, curses.ACS_HLINE, cols) 
        input_win.hline(0, 0, "-", cols) 
        
        # Row 1 and 2 of input_win: Prompt and Buffer
        prompt = "> "
        max_input_width = cols - len(prompt) - 2
        
        input_win.addstr(1, 1, prompt, curses.color_pair(1)) 
        
        display_input = self.input_buffer
        if len(display_input) > max_input_width:
            display_input = "..." + display_input[-(max_input_width - 3):]
            
        input_win.addstr(1, 1 + len(prompt), display_input, curses.color_pair(1))

        # Position the cursor
        cursor_x = 1 + len(prompt) + len(display_input)
        cursor_y = 1
        input_win.move(cursor_y, cursor_x)
        input_win.refresh()


    def _stream_agent_response(self, question, chat_win, rows, cols):
        """Target function for the thread: executes RAG agent streaming."""
        self.streaming_active = True
        self.status = "progress"
        
        self.chat_history.append((self.ragName, ''))
        
        last_msg_index = len(self.chat_history) - 1
        max_chat_rows = rows - self.input_height - self.header_lines

        try:
            for token in self.ragAgent.stream(question):
                current_msg = self.chat_history[last_msg_index][1]
                self.chat_history[last_msg_index] = (self.ragName, current_msg + token)
                
                # AUTO-SCROLLING MECHANISM
                self.scroll_offset = max_chat_rows
                
                self._draw_chat_window(chat_win, max_chat_rows, cols)
                
        except Exception as e:
            error_msg = f"[ERROR] Streaming failed: {e}"
            self.chat_history[last_msg_index] = (self.ragName, error_msg)

        self.streaming_active = False
        self.status = "idle"

        agent_stats = self.ragAgent.getStats()
        self.performance_stats = agent_stats["tokens_per_second"]

        self._draw_chat_window(chat_win, max_chat_rows, cols)
        

    def _handle_key_scrolling(self, key, chat_rows):
        """
        Handles Page Up and Page Down keys for manual scrolling.
        """
        
        # Since the status bar takes one line, the usable content area is chat_rows - 1
        chat_content_rows = chat_rows - 1

        if key == curses.KEY_PPAGE: # Page Up: Show older messages (move view up, decrease offset)
            self.scroll_offset -= chat_content_rows 
            self.scroll_offset = max(0, self.scroll_offset) # Cannot go below 0 (the bottom)
            return True

        elif key == curses.KEY_NPAGE: # Page Down: Show newer messages (move view down, increase offset)
            self.scroll_offset += chat_content_rows
            return True
            
        return False

    def tui_main_loop(self, stdscr):
        """Main curses loop, handles input, redraws, and event processing."""
        
        curses.curs_set(1)
        curses.initscr()
        stdscr.nodelay(True)
        self._init_colors()

        rows, cols = stdscr.getmaxyx()
        
        chat_rows = rows - self.input_height - self.header_lines 
        
        chat_win = curses.newwin(chat_rows, cols, self.header_lines, 0)
        input_win = curses.newwin(self.input_height, cols, rows - self.input_height, 0)
        
        chat_win.keypad(True)
        input_win.keypad(True)

        self.chat_history.append((self.ragName, "Welcome to " + self.ragName + " Agent TUI. Could I help you? Ask me your question."))

        while True:
            # 3.1. Resizing Check
            new_rows, new_cols = stdscr.getmaxyx()
            if new_rows != rows or new_cols != cols:
                rows, cols = new_rows, new_cols
                curses.resizeterm(rows, cols)
                
                chat_rows = rows - self.input_height - self.header_lines
                
                chat_win.resize(chat_rows, cols)
                input_win.mvwin(rows - self.input_height, 0)
                input_win.resize(self.input_height, cols)
                stdscr.clear() 
                self.scroll_offset = 0
            
            # Draw all components
            self._draw_header(stdscr, cols)
            self._draw_chat_window(chat_win, chat_rows, cols)
            self._draw_input_window(input_win, cols)
            
            # 3.2. Input Handling
            c = stdscr.getch()

            if c != -1:
                # Handle scrolling keys
                if self._handle_key_scrolling(c, chat_rows):
                    self._draw_chat_window(chat_win, chat_rows, cols)
                    continue 

                if not self.streaming_active:
                    # Handle typing/Enter keys
                    if c == ord('\n'): # Enter key
                        question = self.input_buffer.strip()
                        if question:
                            if question.lower() in ('q', 'quit', 'exit'):
                                break
                                
                            self.chat_history.append(('You', question))
                            
                            self.scroll_offset = 0 
                            
                            stream_thread = threading.Thread(
                                target=self._stream_agent_response, 
                                args=(question, chat_win, rows, cols)
                            )
                            stream_thread.start()
                            
                            self.input_buffer = ""
                            
                    elif c == curses.KEY_BACKSPACE or c == 127: # Backspace key
                        self.input_buffer = self.input_buffer[:-1]
                        
                    elif 32 <= c <= 126: # Printable character
                        self.input_buffer += chr(c)
            
            # 3.3. Delay
            time.sleep(0.01)

    def run(self):
        """Public method to start the TUI."""
        print("[INFO] Attempting to start Ragnarok TUI. Use 'q' or 'Ctrl+C' to exit.")
        try:
            curses.wrapper(self.tui_main_loop) 
            # Manual screen clear if supported
            try:
                print('\033[H\033[J', end='')
            except Exception:
                pass

        except KeyboardInterrupt:
            print("\n[INFO] TUI interrupted by user (Ctrl+C).")
        except Exception as e:
            print(f"\n[CRITICAL ERROR] TUI failed due to an unexpected error: {e}")

        print("[INFO] Exiting TUI.")

# Usage Example
if __name__ == "__main__":
    tui = CursesTUI()
    tui.run()