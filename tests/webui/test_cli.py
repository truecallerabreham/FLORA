"""Tests for CLI functionality."""

from unittest.mock import Mock, patch

import pytest

from forla.cli._main import _handle_ui_command, main


def test_main_no_command():
    """Test main function with no command provided."""
    with pytest.raises(SystemExit) as exc_info:
        main([])

    assert exc_info.value.code == 1


def test_main_help():
    """Test main function with help flag."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0


def test_main_version():
    """Test main function with version flag."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0


def test_ui_subcommand_help():
    """Test UI subcommand help."""
    with pytest.raises(SystemExit) as exc_info:
        main(["ui", "--help"])

    assert exc_info.value.code == 0


@patch("forla.webui.webui")
def test_ui_subcommand_default_args(mock_webui):
    """Test UI subcommand with default arguments."""
    main(["ui"])

    mock_webui.assert_called_once_with(
        entities_dir=".",
        port=8080,
        host="127.0.0.1",
        auto_open=True,
        reload=False,
        log_level="info",
    )


@patch("forla.webui.webui")
def test_ui_subcommand_custom_args(mock_webui):
    """Test UI subcommand with custom arguments."""
    main(
        [
            "ui",
            "--dir",
            "./agents",
            "--port",
            "8000",
            "--host",
            "0.0.0.0",
            "--no-open",
            "--reload",
            "--log-level",
            "debug",
        ]
    )

    mock_webui.assert_called_once_with(
        entities_dir="./agents",
        port=8000,
        host="0.0.0.0",
        auto_open=False,
        reload=True,
        log_level="debug",
    )


@patch("forla.webui.webui")
def test_handle_ui_command_keyboard_interrupt(mock_webui):
    """Test UI command handling with keyboard interrupt."""
    mock_webui.side_effect = KeyboardInterrupt()

    args = Mock()
    args.dir = "."
    args.port = 8080
    args.host = "127.0.0.1"
    args.no_open = False
    args.reload = False
    args.log_level = "info"

    with pytest.raises(SystemExit) as exc_info:
        _handle_ui_command(args)

    assert exc_info.value.code == 0


@patch("forla.webui.webui")
def test_handle_ui_command_import_error(mock_webui):
    """Test UI command handling with import error."""
    mock_webui.side_effect = ImportError("Module not found")

    args = Mock()
    args.dir = "."
    args.port = 8080
    args.host = "127.0.0.1"
    args.no_open = False
    args.reload = False
    args.log_level = "info"

    with pytest.raises(SystemExit) as exc_info:
        _handle_ui_command(args)

    assert exc_info.value.code == 1


@patch("forla.webui.webui")
def test_handle_ui_command_general_error(mock_webui):
    """Test UI command handling with general error."""
    mock_webui.side_effect = Exception("Something went wrong")

    args = Mock()
    args.dir = "."
    args.port = 8080
    args.host = "127.0.0.1"
    args.no_open = False
    args.reload = False
    args.log_level = "info"

    with pytest.raises(SystemExit) as exc_info:
        _handle_ui_command(args)

    assert exc_info.value.code == 1


def test_main_unknown_command():
    """Test main function with unknown command."""
    with pytest.raises(SystemExit) as exc_info:
        main(["unknown"])

    assert exc_info.value.code == 2
