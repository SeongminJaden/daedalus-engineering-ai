"""CAD embeddings, measured against the two baselines that needed no learning.

The learned embedding is graded SURROGATE and is compared, on the one task the
synthetic set can pose, against the 22 descriptors and the D2 distance
histogram. The numbers in the docstrings are what was measured; the floors
asserted are below them on purpose, because a test that pins a learning curve
to its best day fails on an ordinary one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from brain.semantic.evidence import EvidenceKind
from core.part_dataset import FAMILIES, generate_dataset, make_part
from core.part_dataset.descriptors import describe_step
from core.part_dataset.embedding import (EMBEDDING_DIM, POINTS_PER_PART,
                                         EmbeddingBundle, embedding_label,
                                         nearest_neighbour_precision,
                                         random_rotations, train_embedding)
from core.part_dataset.pointcloud import (canonical_frame, d2_signature,
                                          normalise, point_cloud_of,
                                          sample_surface, tessellate)
from geometry.cad_export.kernel import kernel_available
from nodes import step_analyzer as sa

requires_cad = pytest.mark.skipif(not (kernel_available() and sa.is_available()),
                                  reason="build123d and OCP are required")
FIXTURES = Path("tests/fixtures/cad")

N_TRAIN, N_TEST = 100, 50


def _shape(path):
    contents = sa.read_step(path)
    return contents.shapes[0], contents.unit_to_metres


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    if not (kernel_available() and sa.is_available()):
        pytest.skip("build123d and OCP are required")
    root = tmp_path_factory.mktemp("clouds")
    rng = np.random.default_rng(0)
    out = {}
    for name, n, seed in (("train", N_TRAIN, 1), ("test", N_TEST, 2)):
        records, report = generate_dataset(n, seed=seed, step_dir=root / name,
                                           labelled=False, stop_on_mismatch=True)
        assert report.refused == []
        clouds, d2, desc, fams = [], [], [], []
        for r in records:
            path = root / name / f"{r.part_id}.step"
            shape, unit = _shape(path)
            cloud = point_cloud_of(shape, unit, POINTS_PER_PART, rng)
            clouds.append(cloud)
            d2.append(d2_signature(cloud, rng=rng))
            desc.append(describe_step(path)[0].vector())
            fams.append(r.provenance.generator)
        out[name] = dict(clouds=np.array(clouds), d2=np.array(d2),
                         desc=np.array(desc), fams=fams, records=records)
    return out


@pytest.fixture(scope="module")
def bundle(corpus):
    return train_embedding(corpus["train"]["clouds"], corpus["train"]["fams"],
                           seed=0)


# ------------------------------------------------------------ tessellation

@requires_cad
def test_tessellation_area_matches_the_brep_where_it_can(tmp_path):
    """Exact on planar solids, chorded on curved ones. Measured: relative
    error at round-off on the three planar families, 2e-5 on the plate's
    holes and fillets, up to 8e-4 on the shaft's cylinders."""
    from core.part_dataset import sample_parameters

    rng = np.random.default_rng(3)
    for fam_name, fam in FAMILIES.items():
        params = sample_parameters(fam, rng)
        record, _ = make_part(fam, params, tmp_path, labelled=False)
        shape, unit = _shape(tmp_path / f"{record.part_id}.step")
        mesh = tessellate(shape, unit)
        error = abs(mesh.area_m2 - record.geometry.surface_area_m2) \
            / record.geometry.surface_area_m2
        planar = fam_name in ("box", "hollow_rect", "l_bracket")
        limit = 1e-9 if planar else 2e-3
        assert error < limit, (fam_name, error)


@requires_cad
def test_sampling_is_proportional_to_area(tmp_path):
    """On a box each face gets points in proportion to its area, to within
    the noise of a few thousand draws."""
    fam = FAMILIES["box"]
    params = dict(length_m=0.2, height_m=0.05, width_m=0.02)
    record, _ = make_part(fam, params, tmp_path, labelled=False)
    shape, unit = _shape(tmp_path / f"{record.part_id}.step")
    mesh = tessellate(shape, unit)
    pts = sample_surface(mesh, 20000, np.random.default_rng(1))
    # points on the two large faces (z = +/- 0.01) versus the two small end
    # faces (x = +/- 0.1): areas 0.2*0.05 against 0.05*0.02
    large = np.isclose(np.abs(pts[:, 2]), 0.01, atol=1e-9).sum()
    small = np.isclose(np.abs(pts[:, 0]), 0.10, atol=1e-9).sum()
    assert large / small == pytest.approx((0.2 * 0.05) / (0.05 * 0.02), rel=0.15)


# ------------------------------------------------------------- invariances

def _random_cloud(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(512, 3)) * np.array([3.0, 1.0, 0.3]) \
        + rng.random(3) ** 3          # a skewed cloud with a sign preference


def test_normalise_removes_position_and_size():
    a = _random_cloud()
    b = 7.5 * a + np.array([1.0, -2.0, 3.0])
    assert np.allclose(normalise(a), normalise(b), atol=1e-9)


def test_the_canonical_frame_removes_rotation():
    a = normalise(_random_cloud())
    for rot in random_rotations(5, np.random.default_rng(2)):
        b = a @ rot.T
        assert np.allclose(canonical_frame(a), canonical_frame(b), atol=1e-6)


def test_the_d2_signature_is_invariant_to_similarity_transforms():
    a = normalise(_random_cloud())
    rot = random_rotations(1, np.random.default_rng(4))[0]
    b = 3.0 * (a @ rot.T) + 2.0
    rng_a, rng_b = np.random.default_rng(9), np.random.default_rng(9)
    assert np.allclose(d2_signature(a, rng=rng_a), d2_signature(b, rng=rng_b))
    assert d2_signature(a).sum() == pytest.approx(1.0)


# --------------------------------------------- the learned space, measured

@requires_cad
def test_the_embedding_is_a_unit_vector_and_graded_surrogate(corpus, bundle):
    e = bundle.embed(corpus["test"]["clouds"])
    assert e.shape == (N_TEST, EMBEDDING_DIM)
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)
    item = embedding_label(e[0], "pointnet")
    assert item["evidence"] == "surrogate"
    assert len(item["vector"]) == EMBEDDING_DIM
    d2 = embedding_label(corpus["test"]["d2"][0], "d2_histogram",
                         EvidenceKind.ANALYTICAL)
    assert d2["evidence"] == "simulated"


@requires_cad
def test_the_learned_space_beats_the_histogram_and_not_the_descriptors(corpus,
                                                                        bundle):
    """Measured: descriptors 1.00, PointNet 0.88, D2 0.64. The floors here
    are 0.75 for the learned space and a strict ordering against D2; the
    descriptor result is asserted exactly because the rules already proved
    the topology separates these families."""
    tr, te = corpus["train"], corpus["test"]

    def standardise(x, ref):
        mu, sd = ref.mean(axis=0), ref.std(axis=0)
        sd[sd == 0.0] = 1.0
        return (x - mu) / sd

    p_desc = nearest_neighbour_precision(standardise(te["desc"], tr["desc"]),
                                         te["fams"],
                                         standardise(tr["desc"], tr["desc"]),
                                         tr["fams"])
    p_d2 = nearest_neighbour_precision(te["d2"], te["fams"], tr["d2"],
                                       tr["fams"])
    e_tr, e_te = bundle.embed(tr["clouds"]), bundle.embed(te["clouds"])
    p_learned = nearest_neighbour_precision(e_te, te["fams"], e_tr, tr["fams"])
    print(f"\nprecision at 1: descriptors {p_desc:.2f}  pointnet {p_learned:.2f}"
          f"  d2 {p_d2:.2f}  ({bundle.train_metrics['seconds']:.1f} s on "
          f"{bundle.train_metrics['device']})")
    assert p_desc == 1.0
    assert p_learned >= 0.75
    assert p_learned > p_d2
    assert bundle.train_metrics["final_loss"] < bundle.train_metrics["first_loss"]


@requires_cad
def test_the_same_part_rotated_embeds_the_same(corpus, bundle):
    """Alignment before encoding: measured cosine 1.00 to itself under random
    rotation, against a minimum of 0.19 when the encoder had to learn it."""
    clouds = corpus["test"]["clouds"]
    rots = random_rotations(len(clouds), np.random.default_rng(5))
    rotated = np.array([canonical_frame(c @ r.T) for c, r in zip(clouds, rots)])
    cos = (bundle.embed(clouds) * bundle.embed(rotated)).sum(axis=1)
    assert cos.min() > 0.99


@requires_cad
def test_the_bundle_round_trips(corpus, bundle, tmp_path):
    bundle.save(tmp_path / "emb")
    back = EmbeddingBundle.load(tmp_path / "emb", device=bundle.device)
    a = bundle.embed(corpus["test"]["clouds"][:5])
    b = back.embed(corpus["test"]["clouds"][:5])
    assert np.allclose(a, b, atol=1e-6)
    assert back.families == bundle.families
    assert (tmp_path / "emb" / "meta.json").read_text().count("surrogate") == 1


@requires_cad
@pytest.mark.skipif(not FIXTURES.exists(), reason="the CAD fixtures are not present")
def test_fixtures_land_somewhere_and_the_space_does_not_know_it(corpus, bundle):
    """Every Fusion fixture has a nearest training part. That is all an
    embedding can say: it has no notion of UNKNOWN, which is why the rules
    exist.

    Measured, and it refuted the expectation written here first: the Fusion
    plate A lands nearest a BOX in the learned space, while the rules call it
    a plate. Sampled by area, a 100 by 60 by 10 plate with four 8 mm holes is
    almost entirely slab, and a slab is what the box family also contains.
    The holes that decide the family are a few percent of the surface. So the
    pin is the honest one: the plates land on a slab-like family, and the
    rules, not the embedding, are what say plate.
    """
    from core.part_dataset.classify import rule_classify

    rng = np.random.default_rng(11)
    e_tr = bundle.embed(corpus["train"]["clouds"])
    fams = corpus["train"]["fams"]
    for name in ("fixtureA.step", "fixtureB.step"):
        shape, unit = _shape(FIXTURES / name)
        e = bundle.embed(point_cloud_of(shape, unit, POINTS_PER_PART, rng)[None])[0]
        nearest = fams[int(np.linalg.norm(e_tr - e, axis=1).argmin())]
        print(f"\n{name}: learned nearest {nearest}")
        assert nearest in ("plate_with_holes", "box"), name
        assert rule_classify(describe_step(FIXTURES / name)[0]).family \
            == "plate_with_holes"
