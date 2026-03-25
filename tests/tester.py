import importlib
import json
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from vocoder import audio
from vocoder import core
from vocoder import plot


class HandleArgsTests(unittest.TestCase):
    def test_single_file_path_calls_handle_filename_with_args_only(self):
        args = SimpleNamespace(input_dir='', filename='demo.wav')
        with mock.patch('vocoder.core.handle_filename') as handle_filename:
            core.handle_args(args)
        handle_filename.assert_called_once_with(args)

    def test_vocoder_accepts_filename_as_first_positional_argument(self):
        with mock.patch(
            'vocoder.core.audio.load_audio_file',
            return_value=(np.array([0.1, 0.2, 0.3]), 16000),
        ):
            with mock.patch(
                'vocoder.core.audio.audio_info',
                return_value={
                    'filename': 'examples/1.wav',
                    'n_channels': 1,
                    'sample_rate': 16000,
                    'duration': 3 / 16000,
                },
            ):
                with mock.patch(
                    'vocoder.core.sp.butterworth_bandpass_filter',
                    return_value=np.array([0.1, 0.2, 0.3]),
                ):
                    with mock.patch(
                        'vocoder.core.sp.extract_envelope',
                        return_value=np.array([0.1, 0.2, 0.3]),
                    ):
                        vocoder = core.Vocoder('examples/1.wav')
        self.assertEqual(vocoder.filename, 'examples/1.wav')
        self.assertEqual(vocoder.path, core.Path('examples/1.wav'))

    def test_handle_nbands_uses_default_family_config(self):
        args = SimpleNamespace(
            nbands=6,
            frequency_family='default_family',
            frequency_key=None,
        )
        bands = core.handle_nbands(args)
        np.testing.assert_array_equal(
            bands,
            np.array([50, 229, 558, 1161, 2265, 4290, 7999]),
        )

    def test_handle_nbands_uses_speech_weighted_key(self):
        args = SimpleNamespace(
            nbands=8,
            frequency_family='speech_weighted',
            frequency_key='8_band',
        )
        bands = core.handle_nbands(args)
        np.testing.assert_array_equal(
            bands,
            np.array([50, 180, 350, 600, 950, 1450, 2200, 3500, 7999]),
        )

    def test_vocoder_defaults_to_default_family_six_band(self):
        with mock.patch(
            'vocoder.core.audio.load_audio_file',
            return_value=(np.array([0.1, 0.2, 0.3]), 16000),
        ):
            with mock.patch(
                'vocoder.core.audio.audio_info',
                return_value={
                    'filename': 'examples/1.wav',
                    'n_channels': 1,
                    'sample_rate': 16000,
                    'duration': 3 / 16000,
                },
            ):
                with mock.patch(
                    'vocoder.core.sp.butterworth_bandpass_filter',
                    return_value=np.array([0.1, 0.2, 0.3]),
                ):
                    with mock.patch(
                        'vocoder.core.sp.extract_envelope',
                        return_value=np.array([0.1, 0.2, 0.3]),
                    ):
                        vocoder = core.Vocoder(filename='examples/1.wav')
        np.testing.assert_array_equal(
            vocoder.frequencies,
            np.array([50, 229, 558, 1161, 2265, 4290, 7999]),
        )

    def test_prepare_output_dir_creates_missing_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = core.Path(temp_dir) / 'nested' / 'wav'
            created = core.prepare_output_dir(output_dir)
        self.assertEqual(created, output_dir)

    def test_prepare_output_dir_fails_if_wavs_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = core.Path(temp_dir) / 'wav'
            output_dir.mkdir()
            (output_dir / 'existing.wav').write_bytes(b'RIFF')
            with self.assertRaisesRegex(
                ValueError,
                'already contains wav files',
            ):
                core.prepare_output_dir(output_dir)

    def test_handle_args_searches_input_dir_recursively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = core.Path(temp_dir) / 'input'
            nested_dir = input_dir / 'nested'
            nested_dir.mkdir(parents=True)
            wav_file = nested_dir / 'a.wav'
            wav_file.write_bytes(b'RIFF')
            output_dir = core.Path(temp_dir) / 'output'
            args = SimpleNamespace(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                filename='',
                nprocess=1,
                nbands=6,
                frequency_family='default_family',
                frequency_key=None,
            )
            with mock.patch('vocoder.core.handle_filename') as handle_filename:
                core.handle_args(args)
        self.assertEqual(handle_filename.call_count, 1)
        forwarded_args = handle_filename.call_args[0][0]
        self.assertEqual(forwarded_args.filename, wav_file)

    def test_get_output_filename_uses_hashed_flat_layout(self):
        filename = '/tmp/input/nested/example.wav'
        output_filename = core.get_output_filename(
            filename,
            output_dir='/tmp/output',
            input_dir='/tmp/input',
            n_bands=6,
        )
        self.assertEqual(
            output_filename,
            '/tmp/output/c9ea4a26__example_voc6.wav',
        )

    def test_get_output_filename_adds_shard_directory(self):
        filename = '/tmp/input/nested/example.wav'
        output_filename = core.get_output_filename(
            filename,
            output_dir='/tmp/output',
            input_dir='/tmp/input',
            output_shard_dir='chunk_00001',
            n_bands=6,
        )
        self.assertEqual(
            output_filename,
            '/tmp/output/chunk_00001/c9ea4a26__example_voc6.wav',
        )

    def test_get_output_filename_falls_back_to_flat_output_dir(self):
        filename = '/tmp/input/nested/example.wav'
        output_filename = core.get_output_filename(
            filename,
            output_dir='/tmp/output',
            input_dir='/tmp/other',
            n_bands=4,
        )
        self.assertEqual(
            output_filename,
            '/tmp/output/c361346c__example_voc4.wav',
        )

    def test_build_output_stem_avoids_collisions_for_same_basename(self):
        left = core.build_output_stem('/tmp/input/a/example.wav', '/tmp/input')
        right = core.build_output_stem('/tmp/input/b/example.wav', '/tmp/input')
        self.assertNotEqual(left, right)
        self.assertEqual(left, '76a8cbab__example')
        self.assertEqual(right, 'ad21031f__example')

    def test_legacy_output_to_source_filename_maps_legacy_path(self):
        output_filename = (
            '/scratch-shared/mbentum1/vocoded_bands-6_spidr/wav/'
            'cgn_phrases/N00003_fn000021_104__671-119__144'
            '_vocoded_nbands-6.wav'
        )
        source_filename = core.legacy_output_to_source_filename(
            output_filename,
            legacy_output_dir='/scratch-shared/mbentum1/'
            'vocoded_bands-6_spidr/wav',
            input_dir='/projects/0/prjs1489/data/spidr/wav',
            n_bands=6,
        )
        self.assertEqual(
            source_filename,
            '/projects/0/prjs1489/data/spidr/wav/cgn_phrases/'
            'N00003_fn000021_104__671-119__144.wav',
        )

    def test_legacy_output_to_source_filename_rejects_wrong_suffix(self):
        with self.assertRaisesRegex(
            ValueError,
            'Legacy output does not end with',
        ):
            core.legacy_output_to_source_filename(
                '/scratch-shared/mbentum1/vocoded_bands-6_spidr/wav/'
                'cgn_phrases/example_voc6.wav',
                legacy_output_dir='/scratch-shared/mbentum1/'
                'vocoded_bands-6_spidr/wav',
                input_dir='/projects/0/prjs1489/data/spidr/wav',
                n_bands=6,
            )

    def test_build_output_shard_map_shards_globally(self):
        filenames = [
            core.Path('/tmp/input/flat/a.wav'),
            core.Path('/tmp/input/flat/b.wav'),
            core.Path('/tmp/input/flat/c.wav'),
            core.Path('/tmp/input/nested/d.wav'),
        ]
        shard_map = core.build_output_shard_map(
            filenames,
            '/tmp/input',
            max_files_per_output_dir=2,
        )
        self.assertEqual(
            shard_map,
            {
                '/tmp/input/flat/a.wav': 'chunk_00000',
                '/tmp/input/flat/b.wav': 'chunk_00000',
                '/tmp/input/flat/c.wav': 'chunk_00001',
                '/tmp/input/nested/d.wav': 'chunk_00001',
            },
        )

    def test_append_metadata_writes_jsonl_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = core.Path(temp_dir) / 'meta' / 'batch.jsonl'
            core.append_metadata(
                metadata_path,
                {'input_filename': 'a.wav', 'output_filename': 'b.wav'},
            )
            lines = metadata_path.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(
            lines[0],
            '{"input_filename": "a.wav", "output_filename": "b.wav"}',
        )

    def test_iter_batch_tasks_streams_lightweight_dicts(self):
        filenames = [core.Path('/tmp/input/a.wav'), core.Path('/tmp/input/b.wav')]
        tasks = list(
            core.iter_batch_tasks(
                filenames,
                {'/tmp/input/b.wav': 'chunk_00003'},
            )
        )
        self.assertEqual(
            tasks,
            [
                {
                    'filename': '/tmp/input/a.wav',
                    'output_shard_dir': '',
                    'retry_count': 0,
                },
                {
                    'filename': '/tmp/input/b.wav',
                    'output_shard_dir': 'chunk_00003',
                    'retry_count': 0,
                },
            ],
        )

    def test_make_failure_result_is_json_serializable(self):
        try:
            raise ValueError('broken file')
        except ValueError as exc:
            result = core.make_failure_result(
                'demo.wav',
                np.float32(1.25),
                123,
                exc,
            )
        self.assertEqual(result['status'], 'error')
        self.assertEqual(result['error_type'], 'ValueError')
        self.assertIn('broken file', result['error_message'])
        self.assertIn('ValueError', result['traceback'])
        self.assertIsInstance(result['elapsed_seconds'], float)
        self.assertEqual(result['worker_pid'], 123)

    def test_run_task_subprocess_writes_done_status(self):
        task = {
            'filename': '/tmp/input/example.wav',
            'output_shard_dir': 'chunk_00000',
        }
        worker_config = {
            'sample_rate': 16000,
            'butterworth_order': 4,
            'match_rms': False,
            'output_dir': '/tmp/output',
            'input_dir': '/tmp/input',
            'frequencies': [50, 100, 200],
        }

        class FakeConn:
            def __init__(self):
                self.payload = None
                self.closed = False

            def send(self, value):
                self.payload = value

            def close(self):
                self.closed = True

        class FakeVocoder:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.signal_intensity = 1.25
                self.vocoded_intensity = 2.5
                self.n_bands = 2

            def write_vocoded(self):
                return '/tmp/output/chunk_00000/example_voc2.wav'

        with tempfile.TemporaryDirectory() as temp_dir:
            conn = FakeConn()
            with mock.patch('vocoder.core.Vocoder', FakeVocoder):
                core.run_task_subprocess(
                    task,
                    worker_config,
                    conn,
                    temp_dir,
                    7,
                )
            status_files = list(core.Path(temp_dir).glob('task_00007_pid_*.json'))
            self.assertEqual(len(status_files), 1)
            status = json.loads(status_files[0].read_text())
        self.assertEqual(status['phase'], 'done')
        self.assertEqual(status['input_filename'], '/tmp/input/example.wav')
        self.assertEqual(
            status['output_filename'],
            '/tmp/output/chunk_00000/example_voc2.wav',
        )
        self.assertEqual(conn.payload['status'], 'ok')
        self.assertTrue(conn.closed)

    def test_run_parallel_batch_times_out_stuck_child(self):
        args = SimpleNamespace(
            nprocess=1,
            sample_rate=16000,
            butterworth_order=4,
            match_rms=False,
            output_dir='',
            input_dir='',
            nbands=6,
            frequency_family='default_family',
            frequency_key=None,
            frequencies=None,
            metadata_filename='',
            failure_filename='',
            status_dirname='',
            file_timeout_seconds=0,
        )

        class FakeConn:
            def close(self):
                return None

        class FakeProcess:
            def __init__(self, target, args):
                self.target = target
                self.args = args
                self.pid = 4321
                self.exitcode = None
                self._alive = False

            def start(self):
                self._alive = True

            def is_alive(self):
                return self._alive

            def join(self, timeout = None):
                return None

            def terminate(self):
                self._alive = False
                self.exitcode = -15

            def kill(self):
                self._alive = False
                self.exitcode = -9

        with mock.patch(
            'vocoder.core.multiprocessing.Pipe',
            return_value=(FakeConn(), mock.Mock()),
        ):
            with mock.patch(
                'vocoder.core.multiprocessing.Process',
                side_effect=FakeProcess,
            ):
                with mock.patch('vocoder.core.wait', return_value=[]):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        'Batch failed for 1 files',
                    ):
                        core.run_parallel_batch(
                            args,
                            iter([{'filename': 'stuck.wav'}]),
                            1,
                        )

    def test_run_parallel_batch_retries_timeout_once(self):
        args = SimpleNamespace(
            nprocess=1,
            sample_rate=16000,
            butterworth_order=4,
            match_rms=False,
            output_dir='',
            input_dir='',
            nbands=6,
            frequency_family='default_family',
            frequency_key=None,
            frequencies=None,
            metadata_filename='',
            failure_filename='',
            status_dirname='',
            file_timeout_seconds=0,
        )

        class FakeConn:
            def __init__(self):
                self.recv_calls = 0

            def recv(self):
                self.recv_calls += 1
                return {
                    'status': 'ok',
                    'input_filename': 'retry.wav',
                    'output_filename': 'retry_voc6.wav',
                    'elapsed_seconds': 0.1,
                    'signal_intensity_db': 1.0,
                    'vocoded_intensity_db': 1.0,
                    'n_bands': 6,
                    'worker_pid': 4321,
                }

            def close(self):
                return None

        class FakeProcess:
            pid_counter = 5000

            def __init__(self, target, args):
                self.target = target
                self.args = args
                self.pid = FakeProcess.pid_counter
                FakeProcess.pid_counter += 1
                self.exitcode = None
                self._alive = False

            def start(self):
                self._alive = True

            def is_alive(self):
                return self._alive

            def join(self, timeout = None):
                return None

            def terminate(self):
                self._alive = False
                self.exitcode = -15

            def kill(self):
                self._alive = False
                self.exitcode = -9

        pipe_calls = []

        def fake_pipe(duplex = False):
            conn = FakeConn()
            pipe_calls.append(conn)
            return conn, mock.Mock()

        def fake_wait(conns, timeout = None):
            if len(pipe_calls) < 2 or conns[0] is pipe_calls[0]:
                return []
            return [pipe_calls[1]]

        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = core.Path(temp_dir) / 'retry_voc6.wav'
            output_file.write_bytes(b'partial')
            args.output_dir = temp_dir
            with mock.patch(
                'vocoder.core.multiprocessing.Pipe',
                side_effect=fake_pipe,
            ):
                with mock.patch(
                    'vocoder.core.multiprocessing.Process',
                    side_effect=FakeProcess,
                ):
                    with mock.patch(
                        'vocoder.core.wait',
                        side_effect=fake_wait,
                    ):
                        with mock.patch(
                            'vocoder.core.get_output_filename',
                            return_value=str(output_file),
                        ):
                            core.run_parallel_batch(
                                args,
                                iter([{
                                    'filename': 'retry.wav',
                                    'output_shard_dir': '',
                                    'retry_count': 0,
                                }]),
                                1,
                            )
            self.assertFalse(output_file.exists())

    def test_reloading_core_does_not_import_plot_module(self):
        original_import = __import__

        def guarded_import(name, globals = None, locals = None,
            fromlist = (), level = 0):
            if name == 'vocoder.plot' or name.startswith('matplotlib'):
                raise AssertionError(f'unexpected import: {name}')
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch('builtins.__import__', side_effect=guarded_import):
            reloaded_core = importlib.reload(core)
        self.assertFalse(hasattr(reloaded_core, 'plot'))


class FrequencyBandTests(unittest.TestCase):
    def test_vocoded_signal_uses_envelope_times_band_limited_noise(self):
        parent = SimpleNamespace(
            signal=np.array([0.1, 0.2, 0.3]),
            white_noise=np.array([1.0, 2.0, 3.0]),
            sample_rate=16000,
            butterworth_order=4,
            match_rms=False,
        )
        with mock.patch(
            'vocoder.core.sp.butterworth_bandpass_filter',
            side_effect=[
                np.array([0.5, 0.5, 0.5]),
                np.array([10.0, 20.0, 30.0]),
            ],
        ) as bandpass:
            with mock.patch(
                'vocoder.core.sp.extract_envelope',
                return_value=np.array([2.0, 3.0, 4.0]),
            ):
                band = core.Frequency_band(100, 200, parent)
                np.testing.assert_allclose(
                    band.vocoded_signal,
                    [20.0, 60.0, 120.0],
                )

        np.testing.assert_allclose(band.filtered_signal, [0.5, 0.5, 0.5])
        np.testing.assert_allclose(band.envelope, [2.0, 3.0, 4.0])
        self.assertEqual(bandpass.call_count, 2)


class PlotTests(unittest.TestCase):
    def test_plot_stacked_signals_accepts_missing_envelopes(self):
        with mock.patch('vocoder.plot.plt.subplots') as subplots, \
            mock.patch('vocoder.plot.plot_signal') as plot_signal, \
            mock.patch('vocoder.plot.plt.show'), \
            mock.patch('vocoder.plot.plt.tight_layout'), \
            mock.patch('vocoder.plot.plt.suptitle'):
            subplots.return_value = (mock.Mock(), np.array([mock.Mock(), mock.Mock()]))
            plot.plot_stacked_sigals(
                [np.array([0.0]), np.array([1.0])],
                ['a', 'b'],
                envelopes=None,
            )
        self.assertEqual(plot_signal.call_count, 2)
        self.assertTrue(all(call.kwargs['envelope'] is None for call in plot_signal.call_args_list))

    def test_plot_grid_signals_accepts_missing_envelopes(self):
        axes = np.array([[mock.Mock(), mock.Mock()], [mock.Mock(), mock.Mock()]])
        with mock.patch('vocoder.plot.plt.subplots') as subplots, \
            mock.patch('vocoder.plot.plot_signal') as plot_signal, \
            mock.patch('vocoder.plot.plt.show'), \
            mock.patch('vocoder.plot.plt.tight_layout'), \
            mock.patch('vocoder.plot.plt.suptitle'):
            subplots.return_value = (mock.Mock(), axes)
            plot.plot_grid_signals(
                [np.array([0.0]), np.array([1.0])],
                [np.array([0.0]), np.array([1.0])],
                ['l1', 'l2'],
                ['r1', 'r2'],
                left_side_envelopes=None,
                right_side_envelopes=None,
            )
        self.assertEqual(plot_signal.call_count, 4)
        self.assertTrue(all(call.kwargs['envelope'] is None for call in plot_signal.call_args_list))


class AudioTests(unittest.TestCase):
    def test_play_audio_calls_sounddevice(self):
        fake_sd = mock.Mock()
        with mock.patch('vocoder.audio._load_sounddevice', return_value=fake_sd):
            audio.play_audio(np.array([0.0]))
        fake_sd.play.assert_called_once()
        fake_sd.wait.assert_called_once()

    def test_play_audio_raises_clear_error_without_portaudio(self):
        with mock.patch(
            'vocoder.audio._load_sounddevice',
            side_effect=RuntimeError('Audio playback requires sounddevice'),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                'Audio playback requires sounddevice',
            ):
                audio.play_audio(np.array([0.0]))


if __name__ == '__main__':
    unittest.main()
