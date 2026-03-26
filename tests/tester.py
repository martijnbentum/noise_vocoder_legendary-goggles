import importlib
import io
import json
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

import numpy as np

from vocoder import audio
from vocoder import batch
from vocoder import file_io
from vocoder import plot
from vocoder import slurm_batch
from vocoder import vocoder as vocoder_module


class HandleArgsTests(unittest.TestCase):
    def test_single_file_path_calls_handle_filename_with_args_only(self):
        args = SimpleNamespace(input_dir='', filename='demo.wav')
        with mock.patch('vocoder.batch.handle_filename') as handle_filename:
            batch.handle_args(args)
        handle_filename.assert_called_once_with(args)

    def test_vocoder_accepts_filename_as_first_positional_argument(self):
        with mock.patch(
            'vocoder.vocoder.audio.load_audio_file',
            return_value=(np.array([0.1, 0.2, 0.3]), 16000),
        ):
            with mock.patch(
                'vocoder.vocoder.audio.audio_info',
                return_value={
                    'filename': 'examples/1.wav',
                    'n_channels': 1,
                    'sample_rate': 16000,
                    'duration': 3 / 16000,
                },
            ):
                with mock.patch(
                    'vocoder.vocoder.sp.butterworth_bandpass_filter',
                    return_value=np.array([0.1, 0.2, 0.3]),
                ):
                    with mock.patch(
                        'vocoder.vocoder.sp.extract_envelope',
                        return_value=np.array([0.1, 0.2, 0.3]),
                    ):
                        vocoder = vocoder_module.Vocoder('examples/1.wav')
        self.assertEqual(vocoder.filename, 'examples/1.wav')
        self.assertEqual(
            vocoder.path,
            vocoder_module.Path('examples/1.wav'),
        )

    def test_default_frequency_config_is_packaged_with_module(self):
        self.assertTrue(vocoder_module.FREQUENCY_CONFIG_FILENAME.exists())
        self.assertEqual(
            vocoder_module.FREQUENCY_CONFIG_FILENAME.parent,
            vocoder_module.Path(vocoder_module.__file__).resolve().parent,
        )

    def test_handle_nbands_uses_default_family_config(self):
        args = SimpleNamespace(
            nbands=6,
            frequency_family='default_family',
            frequency_key=None,
        )
        bands = vocoder_module.handle_nbands(args)
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
        bands = vocoder_module.handle_nbands(args)
        np.testing.assert_array_equal(
            bands,
            np.array([50, 180, 350, 600, 950, 1450, 2200, 3500, 7999]),
        )

    def test_vocoder_defaults_to_default_family_six_band(self):
        with mock.patch(
            'vocoder.vocoder.audio.load_audio_file',
            return_value=(np.array([0.1, 0.2, 0.3]), 16000),
        ):
            with mock.patch(
                'vocoder.vocoder.audio.audio_info',
                return_value={
                    'filename': 'examples/1.wav',
                    'n_channels': 1,
                    'sample_rate': 16000,
                    'duration': 3 / 16000,
                },
            ):
                with mock.patch(
                    'vocoder.vocoder.sp.butterworth_bandpass_filter',
                    return_value=np.array([0.1, 0.2, 0.3]),
                ):
                    with mock.patch(
                        'vocoder.vocoder.sp.extract_envelope',
                        return_value=np.array([0.1, 0.2, 0.3]),
                    ):
                        vocoder = vocoder_module.Vocoder(
                            filename='examples/1.wav'
                        )
        np.testing.assert_array_equal(
            vocoder.frequencies,
            np.array([50, 229, 558, 1161, 2265, 4290, 7999]),
        )

    def test_prepare_output_dir_creates_missing_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = vocoder_module.Path(temp_dir) / 'nested' / 'wav'
            created = file_io.prepare_output_dir(output_dir)
        self.assertEqual(created, output_dir)

    def test_prepare_output_dir_fails_if_wavs_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = vocoder_module.Path(temp_dir) / 'wav'
            output_dir.mkdir()
            (output_dir / 'existing.wav').write_bytes(b'RIFF')
            with self.assertRaisesRegex(
                ValueError,
                'already contains wav files',
            ):
                file_io.prepare_output_dir(output_dir)

    def test_handle_args_searches_input_dir_recursively(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = vocoder_module.Path(temp_dir) / 'input'
            nested_dir = input_dir / 'nested'
            nested_dir.mkdir(parents=True)
            wav_file = nested_dir / 'a.wav'
            wav_file.write_bytes(b'RIFF')
            output_dir = vocoder_module.Path(temp_dir) / 'output'
            args = SimpleNamespace(
                input_dir=str(input_dir),
                output_dir=str(output_dir),
                filename='',
                nbands=6,
                frequency_family='default_family',
                frequency_key=None,
            )
            with mock.patch('vocoder.batch.handle_filename') as handle_filename:
                batch.handle_args(args)
        self.assertEqual(handle_filename.call_count, 1)
        forwarded_args = handle_filename.call_args[0][0]
        self.assertEqual(forwarded_args.filename, wav_file)

    def test_get_output_filename_uses_hashed_flat_layout(self):
        filename = '/tmp/input/nested/example.wav'
        output_filename = file_io.get_output_filename(
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
        output_filename = file_io.get_output_filename(
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
        output_filename = file_io.get_output_filename(
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
        left = file_io.build_output_stem('/tmp/input/a/example.wav', '/tmp/input')
        right = file_io.build_output_stem('/tmp/input/b/example.wav', '/tmp/input')
        self.assertNotEqual(left, right)
        self.assertEqual(left, '76a8cbab__example')
        self.assertEqual(right, 'ad21031f__example')

    def test_legacy_output_to_source_filename_maps_legacy_path(self):
        output_filename = (
            '/scratch-shared/mbentum1/vocoded_bands-6_spidr/wav/'
            'cgn_phrases/N00003_fn000021_104__671-119__144'
            '_vocoded_nbands-6.wav'
        )
        source_filename = file_io.legacy_output_to_source_filename(
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
            file_io.legacy_output_to_source_filename(
                '/scratch-shared/mbentum1/vocoded_bands-6_spidr/wav/'
                'cgn_phrases/example_voc6.wav',
                legacy_output_dir='/scratch-shared/mbentum1/'
                'vocoded_bands-6_spidr/wav',
                input_dir='/projects/0/prjs1489/data/spidr/wav',
                n_bands=6,
            )

    def test_build_output_shard_map_shards_globally(self):
        filenames = [
            vocoder_module.Path('/tmp/input/flat/a.wav'),
            vocoder_module.Path('/tmp/input/flat/b.wav'),
            vocoder_module.Path('/tmp/input/flat/c.wav'),
            vocoder_module.Path('/tmp/input/nested/d.wav'),
        ]
        shard_map = file_io.build_output_shard_map(
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
            metadata_path = vocoder_module.Path(temp_dir) / 'meta' / 'batch.jsonl'
            file_io.append_metadata(
                metadata_path,
                {'input_filename': 'a.wav', 'output_filename': 'b.wav'},
            )
            lines = metadata_path.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        self.assertEqual(
            lines[0],
            '{"input_filename": "a.wav", "output_filename": "b.wav"}',
        )

    def test_make_failure_result_is_json_serializable(self):
        try:
            raise ValueError('broken file')
        except ValueError as exc:
            result = batch.make_failure_result(
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

    def test_run_batch_logs_failures_and_continues(self):
        args = SimpleNamespace(
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
            failure_filename='failures.jsonl',
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            args.output_dir = temp_dir
            failure_path = vocoder_module.Path(temp_dir) / 'failures.jsonl'
            results = [
                {
                    'status': 'ok',
                    'input_filename': 'ok.wav',
                    'output_filename': 'ok_voc6.wav',
                    'elapsed_seconds': 0.1,
                    'signal_intensity_db': 1.0,
                    'vocoded_intensity_db': 1.0,
                    'n_bands': 6,
                },
                RuntimeError('broken file'),
            ]

            def fake_handle_filename(args):
                result = results.pop(0)
                if isinstance(result, Exception):
                    raise result
                return result

            with mock.patch(
                'vocoder.batch.handle_filename',
                side_effect=fake_handle_filename,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    'Batch failed for 1 files',
                ):
                    batch.run_batch(
                        args,
                        ['ok.wav', 'broken.wav'],
                        {},
                    )
            lines = failure_path.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        failure = json.loads(lines[0])
        self.assertEqual(failure['input_filename'], 'broken.wav')
        self.assertEqual(failure['error_type'], 'RuntimeError')

    def test_reloading_vocoder_module_does_not_import_plot_module(self):
        original_import = __import__

        def guarded_import(name, globals = None, locals = None,
            fromlist = (), level = 0):
            if name == 'vocoder.plot' or name.startswith('matplotlib'):
                raise AssertionError(f'unexpected import: {name}')
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch('builtins.__import__', side_effect=guarded_import):
            reloaded_module = importlib.reload(vocoder_module)
        self.assertFalse(hasattr(reloaded_module, 'plot'))

    def test_normalize_batch_config_uses_default_band_config(self):
        config = slurm_batch.normalize_batch_config(
            {
                'input_dir': '/tmp/input',
                'output_dir': '/tmp/job/wav',
            }
        )
        self.assertEqual(config['files_per_chunk'], 500)
        self.assertEqual(config['max_parallel_tasks'], 4)
        self.assertFalse(config['match_rms'])
        self.assertEqual(
            config['frequencies'],
            [50, 229, 558, 1161, 2265, 4290, 7999],
        )

    def test_normalize_batch_config_prefers_explicit_frequencies(self):
        config = slurm_batch.normalize_batch_config(
            {
                'input_dir': '/tmp/input',
                'output_dir': '/tmp/job/wav',
                'nbands': 4,
                'frequencies': [10, 20, 40],
            }
        )
        self.assertEqual(config['frequencies'], [10, 20, 40])
        self.assertEqual(config['n_bands'], 2)

    def test_get_run_dir_uses_output_parent(self):
        run_dir = slurm_batch.get_run_dir('/tmp/job/wav')
        self.assertEqual(run_dir, vocoder_module.Path('/tmp/job/_vocode_run'))

    def test_prepare_run_builds_manifest_and_run_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = vocoder_module.Path(temp_dir) / 'input'
            output_dir = vocoder_module.Path(temp_dir) / 'job' / 'wav'
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            (input_dir / 'a.wav').write_bytes(b'RIFF')
            (input_dir / 'b.wav').write_bytes(b'RIFF')
            config_path = vocoder_module.Path(temp_dir) / 'config.json'
            config_path.write_text(json.dumps({
                'input_dir': str(input_dir),
                'output_dir': str(output_dir),
                'files_per_chunk': 1,
            }))
            prepared = slurm_batch.prepare_run(config_path)
            run_paths = slurm_batch.get_run_paths(output_dir)
            manifest_lines = run_paths['manifest'].read_text().splitlines()
            self.assertTrue(run_paths['run_config'].exists())
            self.assertTrue(run_paths['audio_info_dir'].exists())
        self.assertEqual(prepared['total_files'], 2)
        self.assertEqual(prepared['n_chunks'], 2)
        self.assertEqual(prepared['n_task_groups'], 1)
        self.assertEqual(prepared['cpus_per_task'], 16)
        self.assertEqual(len(manifest_lines), 2)

    def test_get_chunk_ids_for_group_groups_chunks_by_worker_count(self):
        chunk_ids = batch.get_chunk_ids_for_group(2, 16, 43)
        self.assertEqual(chunk_ids, list(range(32, 43)))

    def test_get_chunk_output_dir_matches_processing_chunk_id(self):
        self.assertEqual(
            batch.get_chunk_output_dir(42),
            'chunk_00042',
        )

    def test_make_audio_info_record_returns_stem_and_sample_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = vocoder_module.Path(temp_dir) / 'input.wav'
            wav_path = vocoder_module.Path(temp_dir) / 'demo_voc6.wav'
            input_path.write_bytes(b'RIFF')
            wav_path.write_bytes(b'RIFF')
            with mock.patch(
                'vocoder.batch.sf.info',
                return_value=SimpleNamespace(frames=1234),
            ):
                record = batch.make_audio_info_record(
                    wav_path,
                    input_filename=input_path,
                )
        self.assertEqual(record['input_filename'], str(input_path))
        self.assertEqual(record['output_filename'], str(wav_path))
        self.assertEqual(record['file_stem'], 'demo_voc6')
        self.assertEqual(record['n_samples'], 1234)

    def test_count_valid_outputs_uses_processing_chunk_dirs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = vocoder_module.Path(temp_dir) / 'input'
            output_dir = vocoder_module.Path(temp_dir) / 'job' / 'wav'
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            filenames = []
            for index in range(2):
                filename = input_dir / f'{index}.wav'
                filename.write_bytes(b'RIFF')
                filenames.append(str(filename))
            run_dir = output_dir.parent / '_vocode_run'
            manifest_path = run_dir / 'manifest.txt'
            run_dir.mkdir(parents=True)
            manifest_path.write_text('\n'.join(filenames) + '\n')
            output_filename = file_io.get_output_filename(
                filenames[0],
                output_dir=str(output_dir),
                input_dir=str(input_dir),
                output_shard_dir='chunk_00000',
                n_bands=6,
            )
            vocoder_module.Path(output_filename).parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            vocoder_module.Path(output_filename).write_bytes(b'RIFF')
            config = {
                'manifest_path': str(manifest_path),
                'total_files': 2,
                'files_per_chunk': 1,
                'output_dir': str(output_dir),
                'input_dir': str(input_dir),
                'n_bands': 6,
            }
            with mock.patch(
                'vocoder.batch.is_valid_output_file',
                side_effect=lambda path: path == output_filename,
            ):
                completed = slurm_batch.count_valid_outputs(config)
        self.assertEqual(completed, 1)

    def test_slurm_dry_run_prints_grouped_chunk_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = vocoder_module.Path(temp_dir) / 'input'
            output_dir = vocoder_module.Path(temp_dir) / 'job' / 'wav'
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            for index in range(5):
                (input_dir / f'{index}.wav').write_bytes(b'RIFF')
            config_path = vocoder_module.Path(temp_dir) / 'config.json'
            config_path.write_text(json.dumps({
                'input_dir': str(input_dir),
                'output_dir': str(output_dir),
                'files_per_chunk': 2,
                'max_parallel_tasks': 3,
            }))
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                slurm_batch.dry_run(config_path)
        output = buffer.getvalue()
        self.assertIn('total_files: 5', output)
        self.assertIn('n_chunks: 3', output)
        self.assertIn('n_task_groups: 1', output)
        self.assertIn('group=0 chunks=0-2 n_chunks=3 files=5', output)

    def test_process_manifest_chunk_writes_failure_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = vocoder_module.Path(temp_dir) / '_vocode_run'
            manifest_path = run_dir / 'manifest.txt'
            input_dir = vocoder_module.Path(temp_dir) / 'input'
            output_dir = vocoder_module.Path(temp_dir) / 'wav'
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            batch.ensure_run_directories({'run_dir': str(run_dir)})
            input_filename = input_dir / 'broken.wav'
            input_filename.write_bytes(b'RIFF')
            manifest_path.write_text(f'{input_filename}\n')
            config = {
                'run_dir': str(run_dir),
                'manifest_path': str(manifest_path),
                'files_per_chunk': 1,
                'total_files': 1,
                'sample_rate': 16000,
                'match_rms': False,
                'frequencies': [50, 229, 558, 1161, 2265, 4290, 7999],
                'output_dir': str(output_dir),
                'input_dir': str(input_dir),
                'n_bands': 6,
            }
            with mock.patch(
                'vocoder.batch.handle_filename',
                side_effect=RuntimeError('boom'),
            ):
                stats = batch.process_manifest_chunk(config, 0)
            failure_path = batch.get_chunk_failure_path(run_dir, 0)
            lines = failure_path.read_text().splitlines()
        self.assertEqual(stats['failed'], 1)
        self.assertEqual(len(lines), 1)
        failure = json.loads(lines[0])
        self.assertEqual(failure['input_filename'], str(input_filename))
        self.assertEqual(failure['error_type'], 'RuntimeError')


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
            'vocoder.vocoder.sp.butterworth_bandpass_filter',
            side_effect=[
                np.array([0.5, 0.5, 0.5]),
                np.array([10.0, 20.0, 30.0]),
            ],
        ) as bandpass:
            with mock.patch(
                'vocoder.vocoder.sp.extract_envelope',
                return_value=np.array([2.0, 3.0, 4.0]),
            ):
                band = vocoder_module.Frequency_band(100, 200, parent)
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
