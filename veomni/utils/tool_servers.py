"""
Tool server implementations for VeOmni validation.

This module provides various tool server implementations that can be used
during tool validation, including Python execution, bash commands, and
web search capabilities.
"""

import os
import subprocess
import tempfile
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import requests
from urllib.parse import quote_plus
import re
import shutil

from veomni.utils.tool_validation_utils import ToolServer, ToolCallResult

logger = logging.getLogger(__name__)


class PythonToolServer(ToolServer):
    """
    Python code execution tool server.
    
    Executes Python code in a controlled environment with safety checks.
    Based on the verl-tool python_code.py implementation.
    """
    
    def __init__(self, 
                 timeout: int = 10,
                 max_output_length: int = 10000,
                 forbidden_imports: Optional[List[str]] = None,
                 use_firejail: bool = False):
        """
        Initialize the Python tool server.
        
        Args:
            timeout: Maximum execution time in seconds
            max_output_length: Maximum output length to capture
            forbidden_imports: List of forbidden import modules
            use_firejail: Whether to use firejail for sandboxing
        """
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.use_firejail = use_firejail
        
        # Default forbidden imports for security
        self.forbidden_imports = forbidden_imports or [
            'subprocess', 'os', 'sys', 'shutil', 'importlib',
            'eval', 'exec', 'compile', '__import__',
            'open', 'input', 'raw_input'
        ]
        
        # Check if firejail is available
        if self.use_firejail:
            try:
                subprocess.run(['firejail', '--version'], 
                             capture_output=True, check=True)
                self.firejail_available = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("Firejail not available, falling back to regular execution")
                self.firejail_available = False
        else:
            self.firejail_available = False
    
    async def execute_tool(self, tool_name: str, action: str, **kwargs) -> ToolCallResult:
        """Execute Python code."""
        if tool_name.lower() not in ['python', 'python3', 'py']:
            return ToolCallResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' not supported by PythonToolServer",
                tool_name=tool_name
            )
        
        # Extract code from action
        code = self._extract_code(action)
        
        # Security checks
        if not self._is_code_safe(code):
            return ToolCallResult(
                success=False,
                output="",
                error="Code contains forbidden operations",
                tool_name=tool_name
            )
        
        # Execute code
        return await self._execute_code(code, tool_name)
    
    def get_available_tools(self) -> List[str]:
        """Get available tools."""
        return ["python", "python3", "py"]
    
    def _extract_code(self, action: str) -> str:
        """Extract Python code from various formats."""
        # Remove common code block markers
        code = action.strip()
        
        # Handle ```python code blocks
        if code.startswith('```python'):
            code = code[9:]  # Remove ```python
        if code.startswith('```'):
            code = code[3:]  # Remove ```
        if code.endswith('```'):
            code = code[:-3]  # Remove trailing ```
        
        # Handle <python> tags
        if code.startswith('<python>'):
            code = code[8:]
        if code.endswith('</python>'):
            code = code[:-9]
        
        return code.strip()
    
    def _is_code_safe(self, code: str) -> bool:
        """Check if code is safe to execute."""
        # Check for forbidden imports
        for forbidden in self.forbidden_imports:
            if re.search(rf'\b{re.escape(forbidden)}\b', code):
                return False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'__.*__',  # Dunder methods
            r'exec\s*\(',  # exec calls
            r'eval\s*\(',  # eval calls
            r'compile\s*\(',  # compile calls
            r'open\s*\(',  # file operations
            r'input\s*\(',  # input operations
            r'raw_input\s*\(',  # raw input
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        
        return True
    
    async def _execute_code(self, code: str, tool_name: str) -> ToolCallResult:
        """Execute Python code safely."""
        import time
        start_time = time.time()
        
        # Wrap code to capture output
        wrapped_code = f"""
import sys
import io
import traceback

# Capture stdout
old_stdout = sys.stdout
sys.stdout = captured_output = io.StringIO()

# Capture stderr
old_stderr = sys.stderr
sys.stderr = captured_error = io.StringIO()

try:
{chr(10).join('    ' + line for line in code.split(chr(10)))}
except Exception as e:
    print(f"Error: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
finally:
    # Restore stdout/stderr
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    
    # Get captured output
    output = captured_output.getvalue()
    error = captured_error.getvalue()
    
    if error:
        print(f"STDERR: {{error}}")
    if output:
        print(f"STDOUT: {{output}}")
"""
        
        try:
            if self.firejail_available:
                # Use firejail for additional security
                result = await self._execute_with_firejail(wrapped_code)
            else:
                # Execute with subprocess
                result = await self._execute_with_subprocess(wrapped_code)
            
            execution_time = time.time() - start_time
            
            # Parse output
            if result.returncode == 0:
                output = result.stdout.decode('utf-8', errors='replace')
                # Limit output length
                if len(output) > self.max_output_length:
                    output = output[:self.max_output_length] + "... (truncated)"
                
                return ToolCallResult(
                    success=True,
                    output=output,
                    tool_name=tool_name,
                    execution_time=execution_time
                )
            else:
                error = result.stderr.decode('utf-8', errors='replace')
                return ToolCallResult(
                    success=False,
                    output="",
                    error=error,
                    tool_name=tool_name,
                    execution_time=execution_time
                )
        
        except asyncio.TimeoutError:
            return ToolCallResult(
                success=False,
                output="",
                error="Code execution timeout",
                tool_name=tool_name,
                execution_time=self.timeout
            )
        except Exception as e:
            return ToolCallResult(
                success=False,
                output="",
                error=str(e),
                tool_name=tool_name,
                execution_time=time.time() - start_time
            )
    
    async def _execute_with_subprocess(self, code: str) -> subprocess.CompletedProcess:
        """Execute code using subprocess."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute the code
            result = await asyncio.create_subprocess_exec(
                'python3', temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                result.communicate(), 
                timeout=self.timeout
            )
            
            # Create a result object similar to subprocess.CompletedProcess
            class Result:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr
            
            return Result(result.returncode, stdout, stderr)
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except OSError:
                pass
    
    async def _execute_with_firejail(self, code: str) -> subprocess.CompletedProcess:
        """Execute code using firejail for sandboxing."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Execute with firejail
            result = await asyncio.create_subprocess_exec(
                'firejail', '--quiet', '--noprofile', '--noroot',
                '--private-tmp', '--private-dev', '--private-etc',
                '--timeout=10', 'python3', temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                result.communicate(), 
                timeout=self.timeout
            )
            
            class Result:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr
            
            return Result(result.returncode, stdout, stderr)
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except OSError:
                pass


class BashToolServer(ToolServer):
    """
    Bash command execution tool server.
    
    Executes bash commands with safety restrictions.
    """
    
    def __init__(self, 
                 timeout: int = 10,
                 max_output_length: int = 10000,
                 forbidden_commands: Optional[List[str]] = None,
                 use_firejail: bool = False):
        """
        Initialize the bash tool server.
        
        Args:
            timeout: Maximum execution time in seconds
            max_output_length: Maximum output length to capture
            forbidden_commands: List of forbidden commands
            use_firejail: Whether to use firejail for sandboxing
        """
        self.timeout = timeout
        self.max_output_length = max_output_length
        self.use_firejail = use_firejail
        
        # Default forbidden commands for security
        self.forbidden_commands = forbidden_commands or [
            'rm', 'rmdir', 'mv', 'cp', 'chmod', 'chown', 'chgrp',
            'sudo', 'su', 'passwd', 'useradd', 'userdel', 'usermod',
            'mount', 'umount', 'fdisk', 'mkfs', 'fsck',
            'iptables', 'netstat', 'ss', 'lsof',
            'kill', 'killall', 'pkill', 'jobs', 'bg', 'fg',
            'crontab', 'at', 'batch',
            'wget', 'curl', 'nc', 'netcat', 'telnet', 'ssh', 'scp', 'rsync',
            'dd', 'shred', 'wipe'
        ]
    
    async def execute_tool(self, tool_name: str, action: str, **kwargs) -> ToolCallResult:
        """Execute bash command."""
        if tool_name.lower() not in ['bash', 'sh', 'shell']:
            return ToolCallResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' not supported by BashToolServer",
                tool_name=tool_name
            )
        
        # Extract command
        command = self._extract_command(action)
        
        # Security checks
        if not self._is_command_safe(command):
            return ToolCallResult(
                success=False,
                output="",
                error="Command contains forbidden operations",
                tool_name=tool_name
            )
        
        # Execute command
        return await self._execute_command(command, tool_name)
    
    def get_available_tools(self) -> List[str]:
        """Get available tools."""
        return ["bash", "sh", "shell"]
    
    def _extract_command(self, action: str) -> str:
        """Extract bash command from action."""
        command = action.strip()
        
        # Handle ```bash code blocks
        if command.startswith('```bash'):
            command = command[7:]
        if command.startswith('```'):
            command = command[3:]
        if command.endswith('```'):
            command = command[:-3]
        
        # Handle <bash> tags
        if command.startswith('<bash>'):
            command = command[6:]
        if command.endswith('</bash>'):
            command = command[:-7]
        
        return command.strip()
    
    def _is_command_safe(self, command: str) -> bool:
        """Check if command is safe to execute."""
        # Check for forbidden commands
        for forbidden in self.forbidden_commands:
            if re.search(rf'\b{re.escape(forbidden)}\b', command):
                return False
        
        # Check for dangerous patterns
        dangerous_patterns = [
            r'>\s*/dev/',  # Writing to device files
            r'>\s*/etc/',  # Writing to system files
            r'>\s*/usr/',  # Writing to system directories
            r'>\s*/bin/',  # Writing to binary directories
            r'>\s*/sbin/',  # Writing to system binary directories
            r'&\s*$',  # Background processes
            r';\s*&',  # Background processes
            r'\|\s*sh',  # Piping to shell
            r'\|\s*bash',  # Piping to bash
            r'`.*`',  # Command substitution
            r'\$\(',  # Command substitution
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False
        
        return True
    
    async def _execute_command(self, command: str, tool_name: str) -> ToolCallResult:
        """Execute bash command safely."""
        import time
        start_time = time.time()
        
        try:
            # Execute the command
            result = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True
            )
            
            stdout, stderr = await asyncio.wait_for(
                result.communicate(), 
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                output = stdout.decode('utf-8', errors='replace')
                # Limit output length
                if len(output) > self.max_output_length:
                    output = output[:self.max_output_length] + "... (truncated)"
                
                return ToolCallResult(
                    success=True,
                    output=output,
                    tool_name=tool_name,
                    execution_time=execution_time
                )
            else:
                error = stderr.decode('utf-8', errors='replace')
                return ToolCallResult(
                    success=False,
                    output="",
                    error=error,
                    tool_name=tool_name,
                    execution_time=execution_time
                )
        
        except asyncio.TimeoutError:
            return ToolCallResult(
                success=False,
                output="",
                error="Command execution timeout",
                tool_name=tool_name,
                execution_time=self.timeout
            )
        except Exception as e:
            return ToolCallResult(
                success=False,
                output="",
                error=str(e),
                tool_name=tool_name,
                execution_time=time.time() - start_time
            )


class SearchToolServer(ToolServer):
    """
    Web search tool server.
    
    Provides web search capabilities using various search engines.
    """
    
    def __init__(self, 
                 search_engine: str = "duckduckgo",
                 max_results: int = 5,
                 timeout: int = 30):
        """
        Initialize the search tool server.
        
        Args:
            search_engine: Search engine to use ("duckduckgo", "google", "bing")
            max_results: Maximum number of search results
            timeout: Request timeout in seconds
        """
        self.search_engine = search_engine
        self.max_results = max_results
        self.timeout = timeout
        self.session = requests.Session()
    
    async def execute_tool(self, tool_name: str, action: str, **kwargs) -> ToolCallResult:
        """Execute search query."""
        if tool_name.lower() not in ['search', 'google', 'web_search']:
            return ToolCallResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' not supported by SearchToolServer",
                tool_name=tool_name
            )
        
        query = action.strip()
        
        # Execute search
        return await self._execute_search(query, tool_name)
    
    def get_available_tools(self) -> List[str]:
        """Get available tools."""
        return ["search", "google", "web_search"]
    
    async def _execute_search(self, query: str, tool_name: str) -> ToolCallResult:
        """Execute search query."""
        import time
        start_time = time.time()
        
        try:
            if self.search_engine == "duckduckgo":
                results = await self._search_duckduckgo(query)
            else:
                # Fallback to mock search
                results = await self._mock_search(query)
            
            execution_time = time.time() - start_time
            
            # Format results
            formatted_results = self._format_search_results(results)
            
            return ToolCallResult(
                success=True,
                output=formatted_results,
                tool_name=tool_name,
                execution_time=execution_time
            )
        
        except Exception as e:
            return ToolCallResult(
                success=False,
                output="",
                error=str(e),
                tool_name=tool_name,
                execution_time=time.time() - start_time
            )
    
    async def _search_duckduckgo(self, query: str) -> List[Dict[str, str]]:
        """Search using DuckDuckGo."""
        # Simple mock implementation
        # In a real implementation, you would use the DuckDuckGo API
        return [
            {
                "title": f"Mock result 1 for: {query}",
                "url": "https://example.com/1",
                "snippet": f"This is a mock search result for the query '{query}'"
            },
            {
                "title": f"Mock result 2 for: {query}",
                "url": "https://example.com/2", 
                "snippet": f"Another mock result for '{query}'"
            }
        ]
    
    async def _mock_search(self, query: str) -> List[Dict[str, str]]:
        """Mock search implementation."""
        return [
            {
                "title": f"Mock search result for: {query}",
                "url": "https://example.com/mock",
                "snippet": f"This is a mock search result for the query '{query}'"
            }
        ]
    
    def _format_search_results(self, results: List[Dict[str, str]]) -> str:
        """Format search results for display."""
        formatted = []
        for i, result in enumerate(results[:self.max_results], 1):
            formatted.append(f"{i}. {result['title']}")
            formatted.append(f"   URL: {result['url']}")
            formatted.append(f"   {result['snippet']}")
            formatted.append("")  # Empty line between results
        
        return "\n".join(formatted)


class MultiToolServer(ToolServer):
    """
    Multi-tool server that combines multiple tool servers.
    
    This server can route tool calls to appropriate specialized servers.
    """
    
    def __init__(self, tool_servers: Optional[Dict[str, ToolServer]] = None):
        """
        Initialize the multi-tool server.
        
        Args:
            tool_servers: Dictionary of tool name to tool server mappings
        """
        self.tool_servers = tool_servers or {}
        
        # Initialize default tool servers
        if not self.tool_servers:
            self.tool_servers = {
                "python": PythonToolServer(),
                "bash": BashToolServer(),
                "search": SearchToolServer(),
            }
    
    async def execute_tool(self, tool_name: str, action: str, **kwargs) -> ToolCallResult:
        """Execute tool call by routing to appropriate server."""
        # Find the right server for this tool
        server = None
        for server_name, tool_server in self.tool_servers.items():
            if tool_name.lower() in [t.lower() for t in tool_server.get_available_tools()]:
                server = tool_server
                break
        
        if server is None:
            return ToolCallResult(
                success=False,
                output="",
                error=f"No server available for tool '{tool_name}'",
                tool_name=tool_name
            )
        
        return await server.execute_tool(tool_name, action, **kwargs)
    
    def get_available_tools(self) -> List[str]:
        """Get all available tools from all servers."""
        tools = []
        for server in self.tool_servers.values():
            tools.extend(server.get_available_tools())
        return list(set(tools))  # Remove duplicates
    
    def add_tool_server(self, name: str, server: ToolServer):
        """Add a new tool server."""
        self.tool_servers[name] = server
    
    def remove_tool_server(self, name: str):
        """Remove a tool server."""
        if name in self.tool_servers:
            del self.tool_servers[name]