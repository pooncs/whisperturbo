from src.fusion import Fusion
from src.gui import TranslationGUI


def test_gui_refresh_callback_registration():
    """Test that periodic callback is registered in serve method."""
    fusion = Fusion()
    gui = TranslationGUI(fusion)

    # Mock panel state and serve
    import panel as pn

    original_add_periodic_callback = pn.state.add_periodic_callback
    import unittest.mock as mock

    with mock.patch("panel.state.add_periodic_callback") as mock_callback:
        with mock.patch("panel.serve"):
            gui.serve(port=5007)
            # Assert that add_periodic_callback was called with _refresh_table and the correct period
            mock_callback.assert_called_once_with(
                gui._refresh_table,
                period=0.1,  # CONFIG.GUI_REFRESH_RATE / 1000
            )


def test_gui_speaker_filter():
    """Test speaker filter functionality."""
    fusion = Fusion()
    gui = TranslationGUI(fusion)

    # Add some segments
    from src.fusion import TranslatedSegment

    seg1 = TranslatedSegment(0, 1, "text1", "SpeakerA", "en")
    seg2 = TranslatedSegment(1, 2, "text2", "SpeakerB", "en")

    gui.add_segment(seg1)
    gui.add_segment(seg2)

    # Refresh table (will update filter options)
    gui._refresh_table()

    assert "SpeakerA" in gui._speaker_filter.options
    assert "SpeakerB" in gui._speaker_filter.options
