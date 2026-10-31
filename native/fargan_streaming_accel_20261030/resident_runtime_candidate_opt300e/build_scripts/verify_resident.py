"""Check same-ELF repeated requests, cancellation and bounded PCM delivery."""
from pathlib import Path
import argparse
import hashlib
import json
import wave

ROOT = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='resident_han_v1')
    args = parser.parse_args()
    directory = ROOT / 'reports' / args.tag
    run = directory / 'run_001'
    baseline = ROOT.parents[1] / 'opt300b_20261030/resident/reports/resident_final_v1/run_001'
    report = json.loads((directory / 'RESULT.json').read_text())
    assert report['status'] == 'PASSED' and len(report['repeats']) == 1
    metrics = {key: values[0] for key, values in report['repeats'][0]['guest_metrics'].items()}
    assert metrics['runtime_model_init_calls_final'] == 1
    assert metrics['runtime_full_requests'] == 3 and metrics['runtime_cancel_requests'] == 1
    assert metrics['runtime_final_arena_mark'] == metrics['runtime_expected_model_mark']
    assert metrics['arena_failed_requests'] == 0 and metrics['stack_bottom_guard_ok'] == 1
    assert metrics['runtime_PCM_ring_capacity_bytes'] == 3840
    model_mark = metrics['runtime_expected_model_mark']
    expected_names = ['output.pcm', 'output24.pcm', 'condition.f32', 'durations.u32', 'fargan.f32', 'features.bin']
    results = []
    for request_id in (0, 1, 3):
        prefix = f'request_{request_id}_'
        files = {}
        for name in expected_names:
            raw = (run / (prefix+name)).read_bytes()
            reference = (baseline / ('request_'+str(request_id)+'_'+name)).read_bytes()
            assert raw == reference, (request_id, name)
            files[name] = dict(bytes=len(raw), sha256=sha(raw), matches_accepted_baseline=True)
        for name in ['rng.u32', 'gru.f32']:
            raw = (run / (prefix+name)).read_bytes()
            assert raw == (run / ('request_0_'+name)).read_bytes()
            assert raw == (baseline / ('request_'+str(request_id)+'_'+name)).read_bytes()
            files[name] = dict(bytes=len(raw), sha256=sha(raw), matches_request0=True)
        packets = sorted(run.glob(prefix+'packet_*.pcm'), key=lambda path: int(path.stem.split('_')[-1]))
        output = (run / (prefix+'output24.pcm')).read_bytes()
        assert packets and b''.join(path.read_bytes() for path in packets) == output
        assert all(path.stat().st_size == 960 for path in packets[:-1])
        assert 0 < packets[-1].stat().st_size <= 960
        get = lambda name: metrics[f'runtime_request_{request_id}_{name}']
        assert get('model_init_calls') == get('EOS_calls') == 1
        assert get('begin_model_arena_mark') == model_mark
        assert get('clips16') == 0 and get('PCM_ring_peak_packets') <= 4
        assert get('packets') == len(packets) and get('samples24')*2 == len(output)
        rate = metrics['tick_frequency_hz']
        for name, samplerate in [('output', 16000), ('output24', 24000)]:
            with wave.open(str(run / (prefix+name+'.wav')), 'wb') as audio:
                audio.setparams((1, 2, samplerate, 0, 'NONE', 'not compressed'))
                audio.writeframes((run / (prefix+name+'.pcm')).read_bytes())
        results.append(dict(request_id=request_id, files=files, packets=len(packets),
            samples24=get('samples24'), ring_peak_packets=get('PCM_ring_peak_packets'),
            request_arena_end=get('request_arena_end'), arena_peak=get('arena_peak_bytes'),
            first_packet_QEMU_host_seconds=get('first_packet_host_ticks')/rate,
            first_effective_QEMU_host_seconds=get('first_effective_host_ticks')/rate,
            request_QEMU_host_seconds=get('request_finished_host_ticks')/rate,
            RNG_draws=get('RNG_draws'), RNG_next=get('RNG_next'),
            lifetime_weight_ROM_copy_bytes=get('weight_ROM_copy_bytes_lifetime')))
    assert len({row['request_arena_end'] for row in results}) == 1
    assert len({row['arena_peak'] for row in results}) == 1
    assert len({row['lifetime_weight_ROM_copy_bytes'] for row in results}) == 1
    assert metrics['runtime_request_2_cancelled_packets'] == 70
    assert metrics['runtime_request_2_cancel_arena_mark'] == model_mark
    result = dict(status='SAME_ELF_RESIDENT_RUNTIME_REPLAY_AND_CANCEL_VERIFIED',
        elf_sha256=report['elf_sha256'], model_init_calls=1,
        model_init_QEMU_host_seconds=metrics['runtime_model_init_host_ticks']/metrics['tick_frequency_hz'],
        resident_weight_arena_bytes=model_mark, PCM_ring_capacity_bytes=3840,
        full_requests=results, cancelled_request_id=2, cancelled_packets=70,
        final_request_arena_released=True, no_request_arena_growth=True,
        all_accepted_baseline_PCM_condition_durations_byte_identical=True,
        all_complete_request_RNG_and_GRU_bytes_identical=True,
        packed_model_matches_verified_kernel=(ROOT/'assets/model.bin').read_bytes()==(ROOT.parents[1]/'opt300c_20261030/integrated/assets/model.bin').read_bytes(),
        source_model_values_unchanged=True, physical_layout_changed=True,
        prepared_fixed_frontend_features_only=False, dynamic_text_frontend_included=True,
        diagnostic_complete_audio_buffers_retained=True, bounded_PCM_ring_separate_from_diagnostics=True,
        QEMU_host_time_is_not_M55_cycles=True, hardware_300ms_verified=False,
        complete_system_10MB_verified=False, training_updates=0, human_quality_accepted=False,
        memory={key: metrics[key] for key in ['image_code_rodata_and_data_initializer_bytes',
            'assets_qspi_rom_bytes', 'static_rw_bytes', 'arena_reserved_bytes', 'arena_peak_bytes',
            'stack_reserved_bytes', 'stack_written_high_water_bytes', 'reserved_resident_ram_bytes']})
    assert result['packed_model_matches_verified_kernel']
    profile = {key: values for key, values in report['repeats'][0]['guest_metrics'].items() if key.startswith('runtime_total_') or key.startswith('BWE_final_')}
    assert profile and all(len(values)==3 and len(set(values))==1 for values in profile.values()),profile
    assert metrics['FARGAN_model_RAM_zero_copy']==1 and metrics['FARGAN_borrowed_model_RAM_bytes']==833204
    assert metrics['runtime_request_0_weight_ROM_copy_bytes_lifetime']==0
    result['new_profile_fields_reset_across_three_requests']={key:values[0] for key,values in profile.items()}
    result['resident_FARGAN_duplicate_I8_cache_bytes']=0
    result['resident_FARGAN_borrowed_packed_bytes']=833204
    result['backend_ROM_bytes']=result['memory']['image_code_rodata_and_data_initializer_bytes']+result['memory']['assets_qspi_rom_bytes']
    (directory / 'RUNTIME_VERIFIED.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['status', 'model_init_calls',
        'resident_weight_arena_bytes', 'PCM_ring_capacity_bytes', 'no_request_arena_growth']}))


if __name__ == '__main__':
    main()
