def test_public_imports():
    from asianoption import (
        arithmetic_asian_price_mc,
        arithmetic_asian_cv,
        geometric_asian_price_analytical,
        turnbull_wakeman_arithmetic_asian_price,
        levy_arithmetic_asian_price,
        arithmetic_asian_heston_mc,
        arithmetic_asian_sabr_mc,
        heston_effective_vol,
        sabr_effective_vol,
    )

    assert arithmetic_asian_price_mc is not None
    assert arithmetic_asian_cv is not None
    assert geometric_asian_price_analytical is not None
    assert turnbull_wakeman_arithmetic_asian_price is not None
    assert levy_arithmetic_asian_price is not None
    assert arithmetic_asian_heston_mc is not None
    assert arithmetic_asian_sabr_mc is not None
    assert heston_effective_vol is not None
    assert sabr_effective_vol is not None
